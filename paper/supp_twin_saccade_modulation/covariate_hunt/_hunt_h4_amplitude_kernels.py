"""Hunt H4: saccade amplitude/direction-scaled kernels.

Standalone experiment script for the covariate hunt (subagent #4).

Hypothesis: the baseline "both" design treats EVERY microsaccade identically
(``Aadd`` = 0/1 onset lag indicators). Real microsaccades vary in amplitude and
direction; the twin's extraretinal modulation may SCALE with amplitude (or be
direction-tuned). Test whether amplitude-weighted (and direction-tuned) saccade
kernels add INCREMENTAL held-out ``recovered`` over the flat unweighted kernel
already in X_base.

Implementation notes
--------------------
Per-saccade amplitude/direction must line up with the SAME (sacc_trial, sacc_bin)
onsets used by the baseline ``Aadd``. We replay the exact mapping of
``build_saccade_alignment`` (including the buggy per-trial ordering of
``get_inds_from_times``) while carrying the original saccade index, then record
    amplitude_k = hypot(end_x-start_x, end_y-start_y)
    direction_k = atan2(end_y-start_y, end_x-start_x)
for each mapped saccade. The reconstructed (trial,bin) list is asserted equal to
the cached ``align[s]`` lists elementwise before the props are trusted. Cached to
``_hunt_h4_sacc_props.pkl`` so reruns are fast.

Amplitude-weighted lag design (analogous to Aadd, weighted by a per-saccade
scalar w_k):  Aw[tr, b0+lag, li] += w_k  for each mapped saccade k.
Column builders index Aw at rec["tr"], rec["b"] -> (S, L) and optionally multiply
by drive (rec["drive"]) for a gain-form kernel.

Run:
    uv run python paper/supp_twin_saccade_modulation/_hunt_h4_amplitude_kernels.py --build
    uv run python paper/supp_twin_saccade_modulation/_hunt_h4_amplitude_kernels.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation" / "covariate_hunt"))

from _supp_saccade_data import (  # noqa: E402
    load_conditions_cache, build_saccade_alignment, _reconstruct_trial_order,
    DT,
)
from _supp_saccade_model import STA_LAGS, L  # noqa: E402
from _fig3_data import CHECKPOINT_PATH, VALID_TIME_BINS  # noqa: E402

SACC_PROPS_CACHE = CACHE_DIR / "_hunt_h4_sacc_props.pkl"


# ---------------------------------------------------------------------------
# one-time: carry per-saccade amplitude/direction onto the cached alignment
# ---------------------------------------------------------------------------
def build_sacc_props_cache(force=False):
    """Replay build_saccade_alignment carrying per-saccade amplitude/direction.

    Returns {session: {"amp": float[], "dir": float[]}} aligned elementwise with
    align[session]["sacc_trial"]/["sacc_bin"]. Asserts the reconstructed
    (trial,bin) equal the cached alignment before caching.
    """
    if SACC_PROPS_CACHE.exists() and not force:
        print(f"Loading sacc props from {SACC_PROPS_CACHE}")
        with open(SACC_PROPS_CACHE, "rb") as f:
            return dill.load(f)

    import torch
    from DataYatesV1.utils.io import YatesV1Session
    from eval.eval_stack_multidataset import load_model
    from eval.eval_stack_utils import load_single_dataset

    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))

    align = build_saccade_alignment()
    cond = load_conditions_cache()
    cache_by_session = {r["session"]: r for r in cond}

    print(f"Loading model (CPU, no forward) from: {CHECKPOINT_PATH}")
    model, _ = load_model(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    name_to_idx = {n: i for i, n in enumerate(model.names)}

    out = {}
    for session_name, a in align.items():
        rec = cache_by_session[session_name]
        dataset_idx = name_to_idx[session_name]
        print(f"\n--- {session_name} [{dataset_idx}] ---")
        train_data, val_data, _ = load_single_dataset(model, dataset_idx)
        fixrsvp_inds = torch.cat([
            train_data.get_dataset_inds("fixrsvp"),
            val_data.get_dataset_inds("fixrsvp"),
        ], dim=0)
        dset_idx_local = fixrsvp_inds[:, 0].unique().item()
        dset = train_data.dsets[dset_idx_local]

        orig_ids, _eyepos_recon, trial_inds, psth_inds = _reconstruct_trial_order(
            dset, rec["eyepos_used"])
        id_to_pos = {int(t): i for i, t in enumerate(orig_ids)}

        # saccade properties in saccades.json order
        sess = YatesV1Session(session_name)
        saccades = json.load(open(sess.sess_dir / "saccades" / "saccades.json"))
        sacc_times = torch.tensor([s["start_time"] for s in saccades],
                                  dtype=torch.float64)
        amps = np.array([np.hypot(s["end_x"] - s["start_x"],
                                  s["end_y"] - s["start_y"]) for s in saccades])
        dirs = np.array([np.arctan2(s["end_y"] - s["start_y"],
                                    s["end_x"] - s["start_x"]) for s in saccades])

        # replay get_inds_from_times for the target dset ONLY, carrying orig idx.
        # (full function iterates dsets in order then filters to dset_idx_local;
        #  the dset_idx_local block is contiguous & in the same trial order, so
        #  reproducing just this dset's trial loop yields identical row order.)
        t_bins = dset["t_bins"].flatten()
        trials = dset["trial_inds"].flatten()
        unique_trials = torch.unique(trials)
        arange = torch.arange(len(sacc_times))
        ti_parts, orig_parts = [], []
        for iT in unique_trials:
            t_mask = torch.nonzero(trials == iT).flatten()
            trial_times = t_bins[t_mask]
            dt = torch.median(torch.diff(trial_times))
            trial_edges = torch.cat([trial_times - dt / 2,
                                     trial_times[[-1]] + dt / 2])
            tti = torch.bucketize(sacc_times, trial_edges) - 1
            valid = (tti >= 0) & (tti < len(trial_edges) - 1)
            ti_parts.append(t_mask[tti[valid]])
            orig_parts.append(arange[valid])
        time_inds_local = torch.cat(ti_parts).cpu().numpy().astype(int)
        orig_local = torch.cat(orig_parts).cpu().numpy().astype(int)

        # downstream mapping identical to build_saccade_alignment, carry orig idx
        sacc_trial, sacc_bin, amp_list, dir_list = [], [], [], []
        for ti, oi in zip(time_inds_local, orig_local):
            otrial = int(trial_inds[ti])
            pbin = int(psth_inds[ti])
            pos = id_to_pos.get(otrial)
            if pos is None or not (0 <= pbin < VALID_TIME_BINS):
                continue
            sacc_trial.append(pos)
            sacc_bin.append(pbin)
            amp_list.append(float(amps[oi]))
            dir_list.append(float(dirs[oi]))

        # GATE: reconstructed (trial,bin) must equal cached alignment elementwise
        st = np.asarray(sacc_trial, dtype=int)
        sb = np.asarray(sacc_bin, dtype=int)
        assert np.array_equal(st, a["sacc_trial"]), (
            f"{session_name}: sacc_trial mismatch ({len(st)} vs "
            f"{len(a['sacc_trial'])})")
        assert np.array_equal(sb, a["sacc_bin"]), (
            f"{session_name}: sacc_bin mismatch")

        out[session_name] = {"amp": np.asarray(amp_list),
                             "dir": np.asarray(dir_list)}
        print(f"  OK: {len(amp_list)} mapped; amp median "
              f"{np.median(amp_list):.3f} deg, max {np.max(amp_list):.2f}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SACC_PROPS_CACHE, "wb") as f:
        dill.dump(out, f)
    print(f"\nCached sacc props for {len(out)} sessions to {SACC_PROPS_CACHE}")
    return out


# ---------------------------------------------------------------------------
# weighted lag-indicator designs, cached per session on the ctx designs
# ---------------------------------------------------------------------------
def _weighted_lag_design(designs, session, sacc_trial, sacc_bin, weights):
    """Aw[tr, b0+lag, li] += w_k  -- amplitude/direction-weighted analogue of Aadd."""
    d = designs[session]
    T, B = d["Aadd"].shape[0], d["Aadd"].shape[1]
    Aw = np.zeros((T, B, L), dtype=np.float32)
    for tr, b0, w in zip(sacc_trial, sacc_bin, weights):
        for li, lag in enumerate(STA_LAGS):
            b = b0 + lag
            if 0 <= b < B:
                Aw[tr, b, li] += w
    return Aw


def attach_weighted_designs(ctx, props):
    """Precompute the amplitude/direction weighted (T,B,L) designs per session,
    once, and stash them on ctx["designs"][s]. Transforms:
      amp_raw, amp_sqrt, amp_log, amp_ctr (mean-centered raw amp),
      cos (cosθ), sin (sinθ).
    """
    align = build_saccade_alignment()
    designs = ctx["designs"]
    for s in designs:
        if s not in props:
            continue
        a = align[s]
        st, sb = a["sacc_trial"], a["sacc_bin"]
        amp = props[s]["amp"]
        ang = props[s]["dir"]
        transforms = {
            "amp_raw": amp,
            "amp_sqrt": np.sqrt(amp),
            "amp_log": np.log1p(amp),
            "amp_ctr": amp - amp.mean(),          # orthogonal-ish to flat kernel
            "amp_sqrt_ctr": np.sqrt(amp) - np.sqrt(amp).mean(),
            "amp_log_ctr": np.log1p(amp) - np.log1p(amp).mean(),
            "cos": np.cos(ang),
            "sin": np.sin(ang),
        }
        wd = {}
        for name, w in transforms.items():
            wd[name] = _weighted_lag_design(designs, s, st, sb, w)
        designs[s]["_h4"] = wd
    return ctx


# ---------------------------------------------------------------------------
# column builders (index the weighted (T,B,L) designs at rec samples)
# ---------------------------------------------------------------------------
def _rc_lag_basis(n_bumps=4):
    """Raised-cosine bumps over the L saccade lags -> (L, n_bumps) basis.
    Reduces a 37-lag weighted kernel to n_bumps columns (parsimony vs overfit)."""
    lags = np.arange(L, dtype=np.float64)
    centers = np.linspace(0, L - 1, n_bumps)
    width = (L - 1) / (n_bumps - 1) if n_bumps > 1 else L
    B = np.zeros((L, n_bumps))
    for j, c in enumerate(centers):
        d = (lags - c) / width
        B[:, j] = np.where(np.abs(d) < 1.0, 0.5 * (1 + np.cos(np.pi * d)), 0.0)
    return B


_RC = _rc_lag_basis(4)


def make_add_rc(name):
    """Additive amplitude/dir-weighted kernel projected to 4 RC temporal bumps."""
    def builder(rec, designs):
        wd = designs[rec["session"]].get("_h4")
        if wd is None:
            return None
        return wd[name][rec["tr"], rec["b"], :] @ _RC          # (S, 4)
    return builder


def make_mult_rc(name):
    def builder(rec, designs):
        wd = designs[rec["session"]].get("_h4")
        if wd is None:
            return None
        Aw_s = wd[name][rec["tr"], rec["b"], :] @ _RC          # (S, 4)
        return rec["drive"][:, None] * Aw_s
    return builder


def make_add(name):
    """Additive weighted kernel: Aw_s (S, L)."""
    def builder(rec, designs):
        wd = designs[rec["session"]].get("_h4")
        if wd is None:
            return None
        return wd[name][rec["tr"], rec["b"], :]
    return builder


def make_mult(name):
    """Multiplicative (gain) weighted kernel: drive * Aw_s (S, L)."""
    def builder(rec, designs):
        wd = designs[rec["session"]].get("_h4")
        if wd is None:
            return None
        Aw_s = wd[name][rec["tr"], rec["b"], :]
        return rec["drive"][:, None] * Aw_s
    return builder


# ---------------------------------------------------------------------------
def diagnostics(ctx, props):
    """Amplitude distribution + correlation of amp with per-neuron gap-STA magnitude."""
    allamp = np.concatenate([props[s]["amp"] for s in props])
    print("=== amplitude diagnostics (all mapped saccades) ===")
    qs = [1, 5, 25, 50, 75, 95, 99]
    for q, p in zip(qs, np.percentile(allamp, qs)):
        print(f"  amp p{q:02d} = {p:6.3f} deg")
    print(f"  amp max = {allamp.max():.2f} deg;  n = {len(allamp)}")

    # correlation: per mapped saccade, gap at onset (lag 0) vs amplitude, pooled
    # over reliable neurons (magnitude of gap deflection around the saccade).
    designs = ctx["designs"]
    align = build_saccade_alignment()
    li0 = int(np.where(STA_LAGS == 0)[0][0])
    post = (STA_LAGS >= 0) & (STA_LAGS * DT * 1000 <= 100)
    amp_all, mag_all = [], []
    for rec in ctx["reliable_recs"]:
        s = rec["session"]
        if s not in props:
            continue
        a = align[s]
        st, sb = a["sacc_trial"], a["sacc_bin"]
        d = designs[s]
        full = d["full"][:, :, rec["ni"]]
        abl = d["abl"][:, :, rec["ni"]]
        gap = full - abl
        valid = d["dfs"][:, :, rec["ni"]] > 0
        B = d["B"]
        for k, (tr, b0) in enumerate(zip(st, sb)):
            # mean |gap| over 0-100ms post-saccade for this event
            vals = []
            for lag in STA_LAGS[post]:
                b = b0 + lag
                if 0 <= b < B and valid[tr, b]:
                    vals.append(gap[tr, b])
            if vals:
                amp_all.append(props[s]["amp"][k])
                mag_all.append(np.abs(np.mean(vals)))
    amp_all = np.asarray(amp_all)
    mag_all = np.asarray(mag_all)
    if len(amp_all) > 10:
        r = np.corrcoef(amp_all, mag_all)[0, 1]
        r_log = np.corrcoef(np.log1p(amp_all), mag_all)[0, 1]
        print(f"  corr(amp, |post-sacc gap|) = {r:+.4f}  "
              f"(log amp: {r_log:+.4f}, n={len(amp_all)})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true",
                    help="build the sacc-props cache (slow: loads model on CPU)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.build or args.force:
        build_sacc_props_cache(force=args.force)
        print("BUILD DONE")
        if args.build:
            sys.exit(0)

    from _supp_saccade_augment import evaluate_augmentation, load_augment_context

    props = build_sacc_props_cache()
    ctx = load_augment_context()
    attach_weighted_designs(ctx, props)
    diagnostics(ctx, props)

    print("\n=== base check (no extra cols) ===")
    evaluate_augmentation([], "baseline-check", ctx)

    print("\n=== additive amplitude-weighted kernels (over flat Aadd) ===")
    for name in ("amp_raw", "amp_sqrt", "amp_log", "amp_ctr", "amp_sqrt_ctr",
                 "amp_log_ctr"):
        evaluate_augmentation([make_add(name)], f"add {name}", ctx)

    print("\n=== multiplicative amplitude-weighted (gain) kernels ===")
    for name in ("amp_raw", "amp_ctr", "amp_log_ctr"):
        evaluate_augmentation([make_mult(name)], f"mult drive*{name}", ctx)

    print("\n=== amplitude add+mult combos (centered = incremental) ===")
    evaluate_augmentation([make_add("amp_ctr"), make_mult("amp_ctr")],
                          "add+mult amp_ctr", ctx)
    evaluate_augmentation([make_add("amp_log_ctr"), make_mult("amp_log_ctr")],
                          "add+mult amp_log_ctr", ctx)

    print("\n=== direction-tuned kernels (over flat Aadd) ===")
    evaluate_augmentation([make_add("cos"), make_add("sin")], "add cos+sin dir", ctx)
    evaluate_augmentation([make_mult("cos"), make_mult("sin")],
                          "mult drive*(cos+sin) dir", ctx)
    evaluate_augmentation([make_add("cos"), make_add("sin"),
                           make_mult("cos"), make_mult("sin")],
                          "add+mult cos+sin dir", ctx)

    print("\n=== does direction add BEYOND amplitude? ===")
    evaluate_augmentation([make_add("amp_ctr")], "ref: add amp_ctr", ctx)
    evaluate_augmentation([make_add("amp_ctr"), make_add("cos"), make_add("sin")],
                          "add amp_ctr + dir cos/sin", ctx)

    print("\n=== PARSIMONIOUS: 4-bump RC temporal basis (37->4 cols) ===")
    for name in ("amp_ctr", "amp_sqrt_ctr", "amp_log_ctr", "amp_raw"):
        evaluate_augmentation([make_add_rc(name)], f"add-RC {name}", ctx)
    for name in ("amp_ctr", "amp_log_ctr"):
        evaluate_augmentation([make_mult_rc(name)], f"mult-RC drive*{name}", ctx)
    evaluate_augmentation([make_add_rc("amp_ctr"), make_mult_rc("amp_ctr")],
                          "add+mult-RC amp_ctr", ctx)
    evaluate_augmentation([make_add_rc("cos"), make_add_rc("sin")],
                          "add-RC cos+sin dir", ctx)
