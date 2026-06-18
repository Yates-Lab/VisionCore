r"""Figure (Extension 1): mixing time-bin weightings biases the cross-cell
covariance; the consequence is clearest in the shuffle null, and modeling shows
the bias is ADDITIVE, not a multiplicative gain change.

Replaces the old ``fig_time_bin_weighting.py``. Three panels:

  A  variable-n_t staircase + the implicit per-estimator across-bin weights:
     close-pair C_rate is pair-count, McFarland's literal C_psth is uniform 1/T.
     With an envelope alpha(t) co-varying with n_t, E_pair[alpha^2] != E_unif[alpha^2].
  C  SYNTHETIC shuffle-null Dz (truth = 0). A multiplicative GAIN change (the
     alpha(t) envelope) is weighting-invariant -> no bias under mixing. Adding a
     random per-cell ADDITIVE onset transient reproduces the bias: mixed goes
     negative, consistent stays at 0. => the bias is additive, not multiplicative.
  D  REAL shuffle-null Dz: same signature on fixRSVP -- mixed negative,
     consistent ~ 0.

The additive onset transient is included only as one plausible, biologically
realistic candidate for the real-data bias (random per-cell amplitude, with
INDEPENDENT fields -- the bias needs no cross-cell co-tuning), not as a fixture
of the canonical multiplicative generator -- so it is constructed here, post-hoc,
rather than in ``synthetic.make_session``.

Run:  uv run python fig_weighting_bias.py
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from synthetic import make_session
from estimators import (_all_pairs_second_moment,
                        _close_pair_second_moment, _weighted_mean,
                        _weighted_cov, _density_fn)
from VisionCore.covariance import cov_to_corr
from _style import configure, save, C_OK, C_CLOSE, C_TRUTH

HERE = Path(__file__).resolve().parent
REALDATA = HERE / "cache" / "weighting_realdata.pkl"

# colors: consistent (correct) = green, mixed (inconsistent) = red
C_CONS, C_MIX = C_OK, C_CLOSE

NTR, NPH, SIG = 800, 100, 0.15
NT_LO, NT_HI = 15, 360
NCELL = 12
TAMP = 0.30                     # onset-transient amplitude scale (illustrative)
ENV = np.linspace(1.0, 0.05, NPH)        # multiplicative gain envelope alpha(t)
DECAY = np.linspace(1.0, 0.0, NPH)       # additive onset-transient shape


def staircase():
    return np.linspace(NT_HI, NT_LO, NPH).round().astype(int)


# ---------------------------------------------------------------------------
# synthetic regimes (single-bin), mirroring compute_weighting_data definitions
# ---------------------------------------------------------------------------

def _prep(rate, eye):
    counts = np.asarray(rate, float); E_all = np.asarray(eye, float)
    finite = np.isfinite(counts).all(2) & np.isfinite(E_all).all(2)
    keep = finite.sum(0) >= 10; mask = finite & keep[None, :]
    tr, ph = np.where(mask)
    return counts[tr, ph, :], E_all[tr, ph, :], ph


def _offdiag(M):
    iu = np.triu_indices(M.shape[0], 1); return M[iu]


def _fz_off(C):
    r = np.clip(_offdiag(cov_to_corr(C)), -0.999, 0.999); return np.arctanh(r)


def _moments(S, E, T, threshold=0.05):
    phat = _density_fn(E, "gaussian")
    nt_by_t = {t: int((T == t).sum()) for t in np.unique(T)}
    pcw = np.array([max(nt_by_t[t] - 1, 0) / 2.0 for t in T])
    Er_pc = _weighted_mean(S, pcw)
    Er_tr = S.mean(0)
    MM_close = _close_pair_second_moment(S, E, T, phat, "naive", threshold, 1e6,
                                         "pair_count")
    MM_psth_pc = _all_pairs_second_moment(S, E, T, phat, "naive",
                                          time_bin_weighting="pair_count")
    MM_psth_un = _all_pairs_second_moment(S, E, T, phat, "naive",
                                          time_bin_weighting="uniform")
    return dict(pcw=pcw, Er_pc=Er_pc, Er_tr=Er_tr, MM_close=MM_close,
                MM_psth_pc=MM_psth_pc, MM_psth_un=MM_psth_un, phat=phat)


def _regimes(S, E, T, threshold=0.05):
    m = _moments(S, E, T, threshold)
    cons = dict(
        Ctotal=_weighted_cov(S, m["pcw"]),
        Crate=m["MM_close"] - np.outer(m["Er_pc"], m["Er_pc"]),
        Cpsth=m["MM_psth_pc"] - np.outer(m["Er_pc"], m["Er_pc"]),
    )
    mixed = dict(
        Ctotal=np.cov(S.T),
        Crate=m["MM_close"] - np.outer(m["Er_tr"], m["Er_tr"]),
        Cpsth=m["MM_psth_un"] - np.outer(m["Er_tr"], m["Er_tr"]),
    )
    return cons, mixed, m


def _shuffle_eye(E, T, rng):
    Es = E.copy()
    for t in np.unique(T):
        ix = np.where(T == t)[0]
        Es[ix] = E[ix][rng.permutation(len(ix))]
    return Es


def _synth_spikes(seed, add_transient):
    """One independent-field flat-cell session (multiplicative gain envelope),
    optionally with a random per-cell ADDITIVE onset transient, drawn to Poisson
    counts (independent across cells -> true noise correlation 0, and -- with
    independent fields -- ~zero cross-cell PSTH covariance)."""
    nt = staircase()
    sess = make_session(["flat"] * NCELL, n_trials=NTR, n_time_bins=NPH,
                        sigma_eye=SIG, seed=seed, n_trials_per_time_bin=nt,
                        psth_envelope=ENV)
    rate = np.asarray(sess["rate"], float).copy()
    if add_transient:
        rng_a = np.random.default_rng(5000 + seed)
        A = TAMP * rng_a.uniform(0.3, 1.0, NCELL)        # random per-cell amplitude
        rate += A[None, None, :] * DECAY[None, :, None]
    valid = np.isfinite(rate).all(2)
    rate_safe = np.where(np.isfinite(rate), np.clip(rate, 1e-6, None), 0.0)
    rng_p = np.random.default_rng(9000 + seed)
    sp = rng_p.poisson(rate_safe).astype(float)
    sp[~valid] = np.nan
    return sp, sess["eye"]


def _shuffle_dz(S, E, T, rng, threshold=0.05):
    """Off-diagonal shuffle-null Dz for both regimes from one session."""
    cons, mixed, m = _regimes(S, E, T, threshold)
    Es = _shuffle_eye(E, T, rng)
    MM_shuf = _close_pair_second_moment(S, Es, T, m["phat"], "naive", threshold,
                                        1e6, "pair_count")
    Cr_sc = MM_shuf - np.outer(m["Er_pc"], m["Er_pc"])
    Cr_sm = MM_shuf - np.outer(m["Er_tr"], m["Er_tr"])
    dz_c = float(np.nanmean(
        _fz_off(cons["Ctotal"] - Cr_sc) - _fz_off(cons["Ctotal"] - cons["Cpsth"])))
    dz_m = float(np.nanmean(
        _fz_off(mixed["Ctotal"] - Cr_sm) - _fz_off(mixed["Ctotal"] - mixed["Cpsth"])))
    return dz_c, dz_m


def synth_dz_contrast(seeds=range(12)):
    """Panel C: shuffle-null Dz for {gain-only, gain+additive} x {consistent, mixed}.
    Truth is 0; a multiplicative gain leaves both at 0, an additive transient
    drives the mixed estimator negative."""
    out = {"gain": {"cons": [], "mix": []}, "additive": {"cons": [], "mix": []}}
    rng = np.random.default_rng(0)
    for s in seeds:
        for cond, add in (("gain", False), ("additive", True)):
            sp, eye = _synth_spikes(s, add)
            S, E, T = _prep(sp, eye)
            dz_c, dz_m = _shuffle_dz(S, E, T, rng)
            out[cond]["cons"].append(dz_c); out[cond]["mix"].append(dz_m)
    for cond in out:
        for k in out[cond]:
            out[cond][k] = np.array(out[cond][k], float)
    return out


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def _violin(ax, data, positions, colors, points=False, seed=1):
    parts = ax.violinplot(data, positions=positions, showmeans=True, widths=0.7)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col); pc.set_alpha(0.45)
    for key in ("cmeans", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("0.3")
    if points:
        rng = np.random.default_rng(seed)
        for x, d, col in zip(positions, data, colors):
            ax.plot(x + (rng.random(len(d)) - 0.5) * 0.18, d, "o", ms=3,
                    color=col, alpha=0.6)


def main():
    configure()
    with open(REALDATA, "rb") as f:
        rd = pickle.load(f)
    recs = rd["records"]
    real_dz_c = np.array([np.nanmean(r["dz_shuffle_consistent"]) for r in recs])
    real_dz_m = np.array([np.nanmean(r["dz_shuffle_mixed"]) for r in recs])

    contrast = synth_dz_contrast()

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))

    # --- A: staircase + weights ---
    ax = axes[0]
    nt = staircase(); t = np.arange(NPH)
    pair_w = nt * (nt - 1) / 2.0
    ax.bar(t, nt, color="0.8", width=1.0)
    ax.set_xlabel("time-bin index $t$"); ax.set_ylabel(r"$n_t$", color="0.4")
    ax.tick_params(axis="y", labelcolor="0.4")
    ax2 = ax.twinx()
    ax2.plot(t, pair_w / pair_w.sum(), color=C_CONS, lw=1.6,
             label=r"$C_\mathrm{rate}$: pair-count $\propto n_t(n_t{-}1)/2$")
    ax2.plot(t, np.full(NPH, 1.0 / NPH), color=C_MIX, lw=1.6, ls="--",
             label=r"McFarland $C_\mathrm{psth}$: uniform $1/T$")
    ax2.set_ylabel("across-bin weight"); ax2.spines["top"].set_visible(False)
    ax2.legend(loc="upper right", fontsize=7)
    ax.set_title(r"A  variable $n_t$ $\Rightarrow$ $C_\mathrm{rate}$, $C_\mathrm{psth}$"
                 "\nweight bins differently")

    # --- C: synthetic shuffle-null Dz, gain-only vs gain+additive (truth=0) ---
    ax = axes[1]
    data = [contrast["gain"]["cons"], contrast["gain"]["mix"],
            contrast["additive"]["cons"], contrast["additive"]["mix"]]
    _violin(ax, data, [0, 1, 2.5, 3.5], [C_CONS, C_MIX, C_CONS, C_MIX])
    ax.axhline(0, color=C_TRUTH, lw=1.0, ls="--", label="truth = 0")
    ax.set_xticks([0.5, 3.0])
    ax.set_xticklabels(["multiplicative\ngain only", "+ additive\ntransient"])
    ax.set_ylabel(r"shuffle-null $D_z$ (off-diag)")
    ax.set_title("C  synthetic null: additive,\nnot multiplicative (green=cons, red=mix)")
    ax.legend(loc="lower left", fontsize=7)

    # --- D: real shuffle-null Dz ---
    ax = axes[2]
    _violin(ax, [real_dz_c[np.isfinite(real_dz_c)], real_dz_m[np.isfinite(real_dz_m)]],
            [0, 1], [C_CONS, C_MIX], points=True)
    ax.axhline(0, color=C_TRUTH, lw=1.0, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["consistent\n($n_t$)", "mixed\n(McFarland)"])
    ax.set_ylabel(r"shuffle-null $D_z$ (per session)")
    ax.set_title(f"D  real fixRSVP null ({len(recs)} sessions):\nsame fingerprint")

    fig.tight_layout()
    save(fig, "fig_weighting_bias.png")


if __name__ == "__main__":
    main()
