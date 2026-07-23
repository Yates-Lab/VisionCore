"""Hunt H8: time-since-last-saccade / post-saccadic recovery state.

The baseline "both" saccade kernel is a FIXED window STA_LAGS = [-100, +200] ms
(bins -12..+24). Beyond +200 ms it predicts nothing. If the twin carries a slow
post-saccadic gain state (recovery/adaptation outlasting 200 ms, or a gain that
depends CONTINUOUSLY on how long since the last microsaccade), that is unmodeled.

Per sample we build Delta t = bins/ms since the most recent onset at/<= that
(trial, bin) (NaN if none yet in trial), plus inter-saccade interval and local
microsaccade rate. Tested additively, multiplicatively (drive*h), and as an
interaction with the drive-gain axis (#6). NO use of full/y. Causal / input-side.

Run: uv run python paper/supp_twin_saccade_modulation/_hunt_h8_time_since_saccade.py
"""
from __future__ import annotations

import sys
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation" / "covariate_hunt"))

from _supp_saccade_augment import evaluate_augmentation, load_augment_context  # noqa: E402
from _supp_saccade_data import DT  # noqa: E402
from _hunt_h1_drift_velocity import _speed  # noqa: E402

MS = DT * 1000.0  # ms per bin ~8.33


# ---------------------------------------------------------------------------
# per-session Delta-t / ISI / rate arrays (memoized on the designs dict)
# ---------------------------------------------------------------------------
def _sacc_state(d):
    """Return per-session (T,B) arrays, memoized:
       last_onset : bin index of most recent onset <= b (-1 if none in trial yet)
       prev_onset : bin index of the onset BEFORE last_onset (-1 if none)
       csum       : cumulative onset count along bins (for trailing-window rate)
    """
    if "_h8_state" in d:
        return d["_h8_state"]
    T, B = d["T"], d["B"]
    st = np.asarray(d["sacc_trial"], dtype=np.int64)
    sb = np.asarray(d["sacc_bin"], dtype=np.int64)

    # mark[tr,b] = b at onset bins else -1; running max => most recent onset <= b
    mark = np.full((T, B), -1.0)
    mark[st, sb] = sb.astype(float)
    last_onset = np.maximum.accumulate(mark, axis=1)  # (T,B), -1 before first

    # previous onset: for each onset, the onset bin strictly before it in trial.
    # Build by shifting: prev_at_onset[tr, onset_bin] = previous onset bin.
    prev_mark = np.full((T, B), -1.0)
    # sort onsets within trial by bin
    order = np.lexsort((sb, st))
    st_s, sb_s = st[order], sb[order]
    prev_bin = np.full(len(sb_s), -1, dtype=np.int64)
    for i in range(1, len(sb_s)):
        if st_s[i] == st_s[i - 1]:
            prev_bin[i] = sb_s[i - 1]
    prev_mark[st_s, sb_s] = prev_bin.astype(float)
    # forward-fill prev_mark along bins so a sample sees the ISI of its last onset.
    # running "value at last onset": use last_onset index to gather.
    prev_onset = np.full((T, B), -1.0)
    for tr in range(T):
        lo = last_onset[tr]
        pv = prev_mark[tr]
        # for bins where an onset occurred, pv is set; carry forward the pv value
        # associated with the current last_onset bin.
        val = -1.0
        pvcarry = np.full(B, -1.0)
        for b in range(B):
            if mark[tr, b] >= 0:  # onset at this bin
                val = pv[b]
            pvcarry[b] = val
        prev_onset[tr] = pvcarry

    # cumulative onset count along bins for trailing-window rate
    onset_count = np.zeros((T, B))
    np.add.at(onset_count, (st, sb), 1.0)
    csum = np.cumsum(onset_count, axis=1)

    d["_h8_state"] = (last_onset, prev_onset, csum)
    return d["_h8_state"]


def _dt_bins(rec, designs):
    """Delta t in bins since most recent onset <= b; NaN if none yet in trial."""
    d = designs[rec["session"]]
    last_onset, _, _ = _sacc_state(d)
    tr, b = rec["tr"].astype(np.int64), rec["b"].astype(np.int64)
    lo = last_onset[tr, b]
    dt = b.astype(float) - lo
    dt[lo < 0] = np.nan
    return dt  # bins


def _isi_bins(rec, designs):
    """Inter-saccade interval (bins) of the most recent saccade; NaN if undefined."""
    d = designs[rec["session"]]
    last_onset, prev_onset, _ = _sacc_state(d)
    tr, b = rec["tr"].astype(np.int64), rec["b"].astype(np.int64)
    lo = last_onset[tr, b]
    pv = prev_onset[tr, b]
    isi = lo - pv
    isi[(lo < 0) | (pv < 0)] = np.nan
    return isi


def _rate_trailing(rec, designs, win_bins):
    """Count of onsets in trailing window (b-win, b]; local microsaccade rate."""
    d = designs[rec["session"]]
    _, _, csum = _sacc_state(d)
    B = d["B"]
    tr, b = rec["tr"].astype(np.int64), rec["b"].astype(np.int64)
    hi = csum[tr, b]
    blo = b - win_bins
    lo = np.where(blo >= 0, csum[tr, np.clip(blo, 0, B - 1)], 0.0)
    return hi - lo


# ---------------------------------------------------------------------------
# column builders
# ---------------------------------------------------------------------------
def dt_valid(rec, designs):
    """1 where Delta t is defined (a saccade already happened this trial)."""
    dt = _dt_bins(rec, designs)
    return np.isfinite(dt).astype(np.float64)


def make_dt_exp(tau_ms, only_post=None):
    """Additive exp(-Dt/tau) recovery basis. only_post: restrict to Dt>only_post ms
    (isolate the beyond-kernel tail; None = all defined Dt)."""
    def builder(rec, designs):
        dt = _dt_bins(rec, designs)
        dt_ms = dt * MS
        f = np.exp(-dt_ms / tau_ms)
        f = np.where(np.isfinite(dt_ms), f, 0.0)
        if only_post is not None:
            f = np.where(dt_ms > only_post, f, 0.0)
        return f
    return builder


def make_dt_exp_bank(taus, only_post=None):
    def builder(rec, designs):
        dt = _dt_bins(rec, designs)
        dt_ms = dt * MS
        cols = []
        for tau in taus:
            f = np.exp(-dt_ms / tau)
            f = np.where(np.isfinite(dt_ms), f, 0.0)
            if only_post is not None:
                f = np.where(dt_ms > only_post, f, 0.0)
            cols.append(f)
        return np.column_stack(cols)
    return builder


def _rc_time_basis(dt_ms, centers, width):
    """Raised-cosine bumps over Delta-t (ms). undefined Dt -> all zero row."""
    cols = []
    ok = np.isfinite(dt_ms)
    x = np.where(ok, dt_ms, -1e9)
    for c in centers:
        dd = (x - c) / width
        b = np.where(np.abs(dd) < 1.0, 0.5 * (1 + np.cos(np.pi * dd)), 0.0)
        cols.append(np.where(ok, b, 0.0))
    return np.column_stack(cols)


def make_dt_rc(centers, width, only_post=None):
    def builder(rec, designs):
        dt_ms = _dt_bins(rec, designs) * MS
        if only_post is not None:
            dt_ms = np.where(dt_ms > only_post, dt_ms, np.nan)
        return _rc_time_basis(dt_ms, centers, width)
    return builder


def make_mult(fn):
    """Multiplicative gain: drive * f(state)."""
    def builder(rec, designs):
        f = np.asarray(fn(rec, designs), dtype=np.float64)
        if f.ndim == 1:
            f = f[:, None]
        f = np.where(np.isfinite(f), f, 0.0)
        return rec["drive"][:, None] * f
    return builder


# --- #6 drive-gain axis interaction with Dt --------------------------------
def _logdrive(rec):
    return np.log1p(rec["drive"])


def make_gain_x_dt(taus):
    """(drive-gain axis) x h(Dt): does the divisive gain-reduction depend on Dt?
    gain axis ~ log1p(drive) [the #6 static proxy]; multiply by exp recovery."""
    def builder(rec, designs):
        dt_ms = _dt_bins(rec, designs) * MS
        g = _logdrive(rec)
        cols = []
        for tau in taus:
            h = np.exp(-dt_ms / tau)
            h = np.where(np.isfinite(dt_ms), h, 0.0)
            cols.append(g * h)
        return np.column_stack(cols)
    return builder


# --- ISI / rate modulators -------------------------------------------------
def isi_add(rec, designs):
    isi = _isi_bins(rec, designs) * MS
    x = np.where(np.isfinite(isi), np.log1p(isi), 0.0)
    return np.column_stack([x, np.isfinite(isi).astype(float)])


def isi_mult(rec, designs):
    isi = _isi_bins(rec, designs) * MS
    x = np.where(np.isfinite(isi), np.log1p(isi), 0.0)
    return rec["drive"] * x


def make_rate_add(win_ms):
    win_bins = int(round(win_ms / MS))
    def builder(rec, designs):
        return _rate_trailing(rec, designs, win_bins)
    return builder


def make_rate_mult(win_ms):
    win_bins = int(round(win_ms / MS))
    def builder(rec, designs):
        r = _rate_trailing(rec, designs, win_bins)
        return rec["drive"] * r
    return builder


# --- current best model (#1 speed add+mult + #6 drive*[t,t^2]) --------------
def valid_speed(rec, designs):
    return np.isfinite(_speed(rec, designs)).astype(np.float64)


def logspeed_quad(rec, designs):
    sp = _speed(rec, designs)
    x = np.log1p(np.where(np.isfinite(sp), sp, 0.0))
    return np.column_stack([x, x * x])


def mult_logspeed_quad(rec, designs):
    sp = _speed(rec, designs)
    x = np.log1p(np.where(np.isfinite(sp), sp, 0.0))
    return rec["drive"][:, None] * np.column_stack([x, x * x])


def drive_time(rec, designs):
    B = designs[rec["session"]]["B"]
    t = rec["b"].astype(np.float64) / B
    return rec["drive"][:, None] * np.column_stack([t, t * t])


BEST6 = [valid_speed, logspeed_quad, mult_logspeed_quad, drive_time]


# ---------------------------------------------------------------------------
def diagnostics(ctx):
    designs = ctx["designs"]
    recs = ctx["reliable_recs"]
    # pooled: mean-centered gap vs Delta-t bins
    edges_ms = np.array([0, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 700, 1e9])
    n = len(edges_ms) - 1
    sums = np.zeros(n)
    cnts = np.zeros(n)
    frac_def = []
    all_dt = []
    for rec in recs:
        dt_ms = _dt_bins(rec, designs) * MS
        y = rec["y"]
        yc = y - y.mean()
        ok = np.isfinite(dt_ms)
        frac_def.append(ok.mean())
        all_dt.append(dt_ms[ok])
        idx = np.digitize(dt_ms[ok], edges_ms) - 1
        for k in range(n):
            m = idx == k
            if m.any():
                sums[k] += yc[ok][m].sum()
                cnts[k] += m.sum()
    all_dt = np.concatenate(all_dt)
    print("=== H8 diagnostics: time-since-last-saccade (reliable units) ===")
    print(f"fraction of samples with a defined Dt (a saccade already occurred): "
          f"{np.mean(frac_def):.3f}")
    for q in (5, 25, 50, 75, 95, 99):
        print(f"  Dt p{q:02d} = {np.percentile(all_dt, q):7.1f} ms")
    print(f"  Dt max = {all_dt.max():.0f} ms")
    print("\n  pooled mean-centered gap vs Dt bin (post-saccadic recovery shape):")
    labels = ["0-25", "25-50", "50-75", "75-100", "100-150", "150-200",
              "200-250", "250-300", "300-400", "400-500", "500-700", ">700"]
    for k in range(n):
        m = sums[k] / cnts[k] if cnts[k] else np.nan
        print(f"    Dt {labels[k]:>8} ms  n={int(cnts[k]):>8}  mean-gap = {m:+.4f}")

    # a few example units: correlation of gap with Dt beyond +200 ms
    print("\n  per-unit corr(gap, exp(-Dt/150ms)) restricted to Dt>200ms (5 units):")
    for rec in recs[:5]:
        dt_ms = _dt_bins(rec, designs) * MS
        m = np.isfinite(dt_ms) & (dt_ms > 200)
        if m.sum() > 50:
            f = np.exp(-dt_ms[m] / 150.0)
            c = np.corrcoef(rec["y"][m], f)[0, 1]
            print(f"    {rec['session']} ni={rec['ni']:>3}  n(>200ms)={m.sum():>6}  "
                  f"corr={c:+.3f}")


if __name__ == "__main__":
    ctx = load_augment_context()
    diagnostics(ctx)

    print("\n=== R1: baseline reconfirm (0 extra cols must give Δ=0, base 0.4028) ===")
    evaluate_augmentation([], "baseline-check", ctx)

    print("\n=== R1: additive Dt recovery (exp bank) ALONE on saccade base ===")
    evaluate_augmentation([dt_valid, make_dt_exp_bank([100, 200, 400])],
                          "add: exp(-Dt/tau) tau={100,200,400}ms", ctx)
    evaluate_augmentation([dt_valid, make_dt_exp_bank([50, 100, 200, 300, 500])],
                          "add: exp bank tau={50..500}", ctx)
    for tau in (100, 200, 300, 500):
        evaluate_augmentation([dt_valid, make_dt_exp(tau)],
                              f"add: exp(-Dt/{tau}ms) single", ctx)

    print("\n=== R1: additive Dt RC basis over 0-500 ms ===")
    evaluate_augmentation([dt_valid, make_dt_rc([50, 150, 300, 500], 200.0)],
                          "add: RC(Dt) centers 50/150/300/500", ctx)

    print("\n=== R1: BEYOND-kernel tail only (Dt>200ms) ===")
    evaluate_augmentation([dt_valid, make_dt_exp_bank([200, 400], only_post=200)],
                          "add: exp tail Dt>200ms tau={200,400}", ctx)
    evaluate_augmentation([dt_valid, make_dt_rc([300, 450, 650], 250.0, only_post=200)],
                          "add: RC tail Dt>200ms", ctx)

    print("\n=== R2: MULTIPLICATIVE post-saccadic gain drive*h(Dt) ALONE ===")
    evaluate_augmentation([dt_valid, make_mult(make_dt_exp_bank([100, 200, 400]))],
                          "mult: drive*exp(-Dt/tau) tau={100,200,400}", ctx)
    evaluate_augmentation([dt_valid, make_mult(make_dt_rc([50, 150, 300, 500], 200.0))],
                          "mult: drive*RC(Dt)", ctx)

    print("\n=== R2: Dt x drive-gain axis (does #6 divisive gain depend on Dt?) ===")
    evaluate_augmentation([dt_valid, make_gain_x_dt([100, 300, 600])],
                          "log1p(drive)*exp(-Dt/tau) tau={100,300,600}", ctx)

    print("\n=== R2: inter-saccade interval / local rate as gain modulators ===")
    evaluate_augmentation([isi_add], "add: log1p(ISI)+valid", ctx)
    evaluate_augmentation([make_mult(isi_mult)], "mult: drive*log1p(ISI)", ctx)
    for w in (100, 200, 400):
        evaluate_augmentation([make_rate_add(w)], f"add: trailing rate {w}ms", ctx)
    for w in (200, 400):
        evaluate_augmentation([make_rate_mult(w)], f"mult: drive*trailing rate {w}ms", ctx)

    print("\n=== R3: reconfirm best model #6, then STACK H8 on it ===")
    evaluate_augmentation(BEST6, "BEST6 (#1 speed + #6 drive*[t,t^2])", ctx)
    evaluate_augmentation(BEST6 + [dt_valid, make_dt_exp_bank([100, 200, 400])],
                          "BEST6 + add exp(Dt) bank", ctx)
    evaluate_augmentation(BEST6 + [dt_valid, make_mult(make_dt_exp_bank([100, 200, 400]))],
                          "BEST6 + mult drive*exp(Dt) bank", ctx)
    evaluate_augmentation(BEST6 + [dt_valid, make_gain_x_dt([100, 300, 600])],
                          "BEST6 + log1p(drive)*exp(Dt)", ctx)
    evaluate_augmentation(BEST6 + [make_rate_mult(200)],
                          "BEST6 + mult drive*trailing rate 200ms", ctx)
