"""Hunt H10 (FINAL): completeness + synthesis pass for the variance hunt.

NOT a new marginal hypothesis. Four jobs:
 1. Reconfirm the #9 headline model (base 0.4028 -> aug 0.4201, +0.0108 paired,
    87% units up); per-neuron recovered distribution; per-subject / per-session
    breakdown.
 2. Completeness critic: residual res = y - yhat_headline is NOT low-D
    recoverable. Augment the HEADLINE (not the "both" base) with each remaining
    candidate signal and report the INCREMENTAL Δ over the headline, guarding the
    paired-median artifact (Δ-median up but aug-median down = overfit, not gain).
 3. Where does the +0.0108 concentrate: distribution of per-unit Δ, and whether
    high-Δ units share a property (drift-speed variance, drive, gap variance).
 4. Honest ceiling statement.

Headline (6 aug cols on the "both" saccade base):
    [speed_valid, logspeed, logspeed^2, drive*t, drive*t^2, drive*min(speed,10)]

Run: uv run python paper/supp_twin_saccade_modulation/_hunt_h10_completeness.py
"""
from __future__ import annotations

import sys
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"))

from _supp_saccade_augment import (  # noqa: E402
    evaluate_augmentation, load_augment_context, cols_time_in_trial,
)
from _supp_saccade_data import DT  # noqa: E402
from _hunt_h6_drive_gain import (  # noqa: E402
    speed_valid, logspeed_quad, drive_x_time, logdrive_quad, drive_cubic,
)
from _hunt_h9_joint_surface import make_drive_x_speed_cap  # noqa: E402

# ---------------------------------------------------------------------------
# THE HEADLINE MODEL (exact #9 winning columns) -- 6 aug cols on "both" base
# ---------------------------------------------------------------------------
HEADLINE = [speed_valid, logspeed_quad, drive_x_time, make_drive_x_speed_cap(10.0)]


# ---------------------------------------------------------------------------
# completeness-critic candidate builders (all input-side, no leakage)
# ---------------------------------------------------------------------------
EXT_LAGS = np.array([-22, -20, -18, -16, -14,
                     26, 28, 30, 32, 34, 36, 40, 44, 48])   # outside base window
W_RATE = 24  # 200 ms trailing window for microsaccade rate


def attach_extras(designs):
    """Precompute per-session extended-window onset kernel, per-trial onset lists,
    and trailing microsaccade rate. Runs once."""
    for s, d in designs.items():
        T, B = d["T"], d["B"]
        st = np.asarray(d["sacc_trial"], dtype=np.int64)
        sb = np.asarray(d["sacc_bin"], dtype=np.int64)
        # extended onset kernel (beyond the -100..+200 ms base window)
        Aext = np.zeros((T, B, len(EXT_LAGS)), dtype=np.float32)
        for tr, b0 in zip(st, sb):
            for li, lag in enumerate(EXT_LAGS):
                b = b0 + lag
                if 0 <= b < B:
                    Aext[tr, b, li] += 1.0
        d["Aext"] = Aext
        # per-trial sorted onset bins (for time-since-last-saccade)
        onsets = [[] for _ in range(T)]
        for tr, b0 in zip(st, sb):
            onsets[tr].append(b0)
        d["onsets"] = [np.sort(np.array(o, dtype=np.int64)) if o
                       else np.array([], dtype=np.int64) for o in onsets]
        # trailing microsaccade rate (count of onsets in last W_RATE bins)
        rate = np.zeros((T, B), dtype=np.float32)
        for tr in range(T):
            arr = d["onsets"][tr]
            if len(arr) == 0:
                continue
            hist = np.bincount(arr, minlength=B)[:B]
            cum = np.cumsum(hist)
            lagged = np.concatenate([np.zeros(W_RATE), cum])[:B]
            rate[tr] = cum - lagged
        d["msrate"] = rate
    return designs


# (a) eye position gain field: [x, y, x^2, y^2] + validity indicator
def eyepos_quad(rec, designs):
    ep = designs[rec["session"]]["eyepos"][rec["tr"], rec["b"], :]  # (S,2)
    x, y = ep[:, 0], ep[:, 1]
    valid = np.isfinite(x).astype(np.float64)
    return np.column_stack([x, y, x * x, y * y, valid])


# (b) time-since-last-saccade basis: exp(-dt/tau) bank + validity
def time_since_sacc(rec, designs):
    s = rec["session"]
    onsets = designs[s]["onsets"]
    tr = rec["tr"]
    b = rec["b"].astype(np.int64)
    dt_ms = np.full(len(tr), np.nan)
    for i in range(len(tr)):
        arr = onsets[int(tr[i])]
        if arr.size == 0:
            continue
        j = np.searchsorted(arr, b[i], side="right") - 1
        if j >= 0:
            dt_ms[i] = (b[i] - arr[j]) * DT * 1000.0
    valid = np.isfinite(dt_ms).astype(np.float64)
    dtf = np.where(np.isfinite(dt_ms), dt_ms, 0.0)
    cols = [valid]
    for tau in (100.0, 200.0, 400.0):
        cols.append(valid * np.exp(-dtf / tau))
    return np.column_stack(cols)


# (c) finer/extended saccade kernel: onset indicators outside the base window,
#     additive + drive-scaled (multiplicative)
def ext_kernel_add(rec, designs):
    return designs[rec["session"]]["Aext"][rec["tr"], rec["b"], :]  # (S,K)


def ext_kernel_mult(rec, designs):
    A = designs[rec["session"]]["Aext"][rec["tr"], rec["b"], :]
    return rec["drive"][:, None] * A


# (d) time-in-trial [t, t^2] standalone -> use harness cols_time_in_trial

# (e) local microsaccade rate (trailing 200 ms), additive + drive-scaled
def msrate_add(rec, designs):
    r = designs[rec["session"]]["msrate"][rec["tr"], rec["b"]]
    return r.astype(np.float64)


def msrate_mult(rec, designs):
    r = designs[rec["session"]]["msrate"][rec["tr"], rec["b"]].astype(np.float64)
    return np.column_stack([r, rec["drive"].astype(np.float64) * r])


# (f) drive higher orders (headline has NO pure static-drive term)
def drive_higher(rec, designs):
    d = rec["drive"].astype(np.float64)
    ld = np.log1p(d)
    return np.column_stack([ld, ld * ld, ld * ld * ld])


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------
def _iqr(x):
    x = x[np.isfinite(x)]
    return np.percentile(x, 25), np.percentile(x, 75)


def incremental_over_headline(res_head, res_full, label, out):
    """Per-neuron incremental recovered of (headline+cand) over headline.
    Guards the paired-median artifact: report BOTH paired-Δ-median and aug-median."""
    a = res_head["aug"]
    b = res_full["aug"]
    d = b - a
    fin = np.isfinite(d)
    med_d = float(np.nanmedian(d))
    mean_d = float(np.nanmean(d))
    frac_up = float(np.mean(d[fin] > 0))
    npos = int((d[fin] > 0).sum())
    nneg = int((d[fin] < 0).sum())
    aug_med = res_full["med_aug"]
    head_med = res_head["med_aug"]
    artifact = " <ARTIFACT: aug-median FELL>" if (med_d > 0 and aug_med < head_med) else ""
    line = (f"  [{label:38s}] incr Δmed {med_d:+.4f} (mean {mean_d:+.4f}) | "
            f"aug-med {head_med:.4f}->{aug_med:.4f} | {frac_up*100:3.0f}% up "
            f"(+{npos}/-{nneg}){artifact}")
    print(line)
    out.append(line)
    return d


def main():
    out = []

    def emit(s=""):
        print(s)
        out.append(s)

    ctx = load_augment_context()
    attach_extras(ctx["designs"])
    recs = ctx["reliable_recs"]
    emit(f"reliable neurons: {len(recs)}")

    # ================= TASK 1: reconfirm headline =========================
    emit("\n" + "=" * 74)
    emit("TASK 1 - RECONFIRM HEADLINE")
    emit("=" * 74)
    r_base = evaluate_augmentation([], "baseline-check (no extra cols)", ctx, verbose=False)
    emit(r_base["summary"])
    r_head = evaluate_augmentation(HEADLINE, "HEADLINE (#9 6-col)", ctx, verbose=False)
    emit(r_head["summary"])

    base, aug, delta = r_base["base"], r_head["aug"], r_head["delta"]
    # per-neuron recovered distribution
    b25, b75 = _iqr(base)
    a25, a75 = _iqr(aug)
    emit(f"\n  per-neuron recovered: BASE   median {np.nanmedian(base):.4f}  "
         f"IQR [{b25:.4f}, {b75:.4f}]")
    emit(f"  per-neuron recovered: HEAD   median {np.nanmedian(aug):.4f}  "
         f"IQR [{a25:.4f}, {a75:.4f}]")

    # per-subject
    subj = np.array([r["subject"] for r in recs])
    emit("\n  per-subject median recovered (base -> head, paired Δ-median):")
    for sname in sorted(set(subj)):
        m = subj == sname
        n = int(m.sum())
        emit(f"    {sname:8s} N={n:4d}: base {np.nanmedian(base[m]):.4f} -> "
             f"head {np.nanmedian(aug[m]):.4f}  (Δmed {np.nanmedian(delta[m]):+.4f}, "
             f"{np.mean(delta[m][np.isfinite(delta[m])]>0)*100:.0f}% up)")

    # per-session spread
    sess = np.array([r["session"] for r in recs])
    sess_head = []
    sess_delta = []
    for s in sorted(set(sess)):
        m = sess == s
        if m.sum() >= 5:
            sess_head.append(np.nanmedian(aug[m]))
            sess_delta.append(np.nanmedian(delta[m]))
    sess_head = np.array(sess_head)
    sess_delta = np.array(sess_delta)
    emit(f"\n  per-session (>=5 units, n={len(sess_head)} sessions) median HEAD "
         f"recovered: p25/50/75 = "
         f"{np.percentile(sess_head,25):.3f}/{np.percentile(sess_head,50):.3f}/"
         f"{np.percentile(sess_head,75):.3f}  range [{sess_head.min():.3f},"
         f"{sess_head.max():.3f}]")
    emit(f"  per-session median Δ: p25/50/75 = "
         f"{np.percentile(sess_delta,25):+.4f}/{np.percentile(sess_delta,50):+.4f}/"
         f"{np.percentile(sess_delta,75):+.4f}  ({np.mean(sess_delta>0)*100:.0f}% "
         f"of sessions positive)")

    # ================= TASK 2: completeness critic ========================
    emit("\n" + "=" * 74)
    emit("TASK 2 - COMPLETENESS CRITIC (incremental over HEADLINE; residual test)")
    emit("=" * 74)
    emit("  Does ANY remaining signal linearly predict the headline residual?")
    emit("  (augment headline; watch <ARTIFACT> = Δ-median up but aug-median down)\n")

    candidates = [
        ("a: eyepos [x,y,x2,y2]+valid", [eyepos_quad]),
        ("b: time-since-sacc exp bank", [time_since_sacc]),
        ("c: ext saccade kernel (add)", [ext_kernel_add]),
        ("c: ext saccade kernel (add+mult)", [ext_kernel_add, ext_kernel_mult]),
        ("d: time-in-trial [t,t2]", [cols_time_in_trial]),
        ("e: microsacc rate (add)", [msrate_add]),
        ("e: microsacc rate (add+mult)", [msrate_mult]),
        ("f: log-drive higher orders", [drive_higher]),
    ]
    for label, cand in candidates:
        r_full = evaluate_augmentation(HEADLINE + cand, "H+" + label, ctx, verbose=False)
        incremental_over_headline(r_head, r_full, label, out)

    # kitchen-sink: ALL candidates at once (max chance for anything to survive)
    all_cands = [eyepos_quad, time_since_sacc, ext_kernel_add, ext_kernel_mult,
                 cols_time_in_trial, msrate_mult, drive_higher]
    r_sink = evaluate_augmentation(HEADLINE + all_cands, "H+kitchen-sink", ctx, verbose=False)
    emit("")
    incremental_over_headline(r_head, r_sink, "ALL candidates (kitchen sink)", out)

    # ================= TASK 3: where does variance concentrate =============
    emit("\n" + "=" * 74)
    emit("TASK 3 - WHERE DOES THE +0.0108 CONCENTRATE?")
    emit("=" * 74)
    d = delta.copy()
    fin = np.isfinite(d)
    df = d[fin]
    emit(f"  per-unit Δ distribution (n={fin.sum()}): "
         f"p10 {np.percentile(df,10):+.4f}  p25 {np.percentile(df,25):+.4f}  "
         f"p50 {np.percentile(df,50):+.4f}  p75 {np.percentile(df,75):+.4f}  "
         f"p90 {np.percentile(df,90):+.4f}")
    emit(f"  mean Δ {df.mean():+.4f}; {np.mean(df>0)*100:.0f}% units up; "
         f"top-10% units hold {np.sort(df)[::-1][:int(0.1*len(df))].sum()/df.sum()*100:.0f}% "
         f"of total positive-sum Δ")

    # per-unit properties vs Δ
    designs = ctx["designs"]
    sp_var = np.full(len(recs), np.nan)
    sp_p95 = np.full(len(recs), np.nan)
    drive_mean = np.full(len(recs), np.nan)
    y_var = np.full(len(recs), np.nan)
    for i, r in enumerate(recs):
        sp = designs[r["session"]]["speed"][r["tr"], r["b"]]
        spf = sp[np.isfinite(sp)]
        if spf.size > 10:
            sp_var[i] = np.var(spf)
            sp_p95[i] = np.percentile(spf, 95)
        drive_mean[i] = np.mean(r["drive"])
        y_var[i] = np.var(r["y"])

    def _corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 20:
            return np.nan
        return np.corrcoef(a[m], b[m])[0, 1]

    emit("\n  corr(per-unit Δ, property) across reliable units:")
    emit(f"    drift-speed variance : {_corr(d, sp_var):+.3f}")
    emit(f"    drift-speed p95      : {_corr(d, sp_p95):+.3f}")
    emit(f"    mean drive           : {_corr(d, drive_mean):+.3f}")
    emit(f"    gap (y) variance     : {_corr(d, y_var):+.3f}")
    emit(f"    baseline recovered   : {_corr(d, base):+.3f}")

    # high-Δ vs low-Δ unit property contrast (top vs bottom quartile of Δ)
    q1, q3 = np.percentile(df, 25), np.percentile(df, 75)
    hi = fin & (d >= q3)
    lo = fin & (d <= q1)
    emit("\n  top-Δ quartile vs bottom-Δ quartile (median property):")
    emit(f"    drift-speed var : hi {np.nanmedian(sp_var[hi]):7.2f}  "
         f"lo {np.nanmedian(sp_var[lo]):7.2f}")
    emit(f"    drift-speed p95 : hi {np.nanmedian(sp_p95[hi]):7.2f}  "
         f"lo {np.nanmedian(sp_p95[lo]):7.2f}")
    emit(f"    mean drive      : hi {np.nanmedian(drive_mean[hi]):7.2f}  "
         f"lo {np.nanmedian(drive_mean[lo]):7.2f}")

    # ================= TASK 4: honest ceiling =============================
    emit("\n" + "=" * 74)
    emit("TASK 4 - HONEST CEILING")
    emit("=" * 74)
    kernel = float(np.nanmedian(base))
    head = float(np.nanmedian(aug))
    paired = r_head["med_delta"]
    emit(f"  saccade kernel recovers        : {kernel*100:.1f}% of gap variance (median unit)")
    emit(f"  + interpretable gain terms     : aug-median {head*100:.1f}%  "
         f"(paired Δ-median {paired*100:+.2f} pts)")
    emit(f"  remaining, unrecovered by any")
    emit(f"    simple marginal function     : ~{(1-head)*100:.0f}% of gap variance")
    emit(f"  => distributed across the twin's nonlinear recurrent computation.")

    # write to file
    fpath = (VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"
             / "_hunt_h10_output.txt")
    fpath.write_text("\n".join(out) + "\n")
    print(f"\n[written to {fpath}]")


if __name__ == "__main__":
    main()
