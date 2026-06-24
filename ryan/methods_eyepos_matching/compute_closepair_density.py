r"""Directly-estimated close-pair density vs the squared-marginal assumption.

The matched estimator (writeup §4.2/§4.4) reweights toward a target eye
distribution using importance weights that assume the close-pair midpoint
density equals ``p_hat^2`` -- the §A.5 single-bin ``Δe → 0`` identity. In the
production trajectory-mode estimator each sample's eye trajectory is reduced to
a geometric-median representative point ``ρ`` and "close" is judged by the
whole-window RMS distance, so that identity is only approximate. This driver
quantifies the gap two ways, on the same 25 real ``fixRSVP`` sessions and the
exact §4.5 estimator window:

  1. **Density comparison.** Per session, fit ``p_hat`` on the representative
     points ``{ρ_i}`` and ``p_hat_pair`` directly on the realized close-pair
     representative-midpoints ``{½(ρ_i+ρ_j)}``, and measure how far
     ``p_hat_pair`` is from ``p_hat^2``: the close-pair variance ratio
     (``tr cov(ρ_mid) / tr cov(ρ)``, ideal 0.5 for a Gaussian) and
     ``KL(p_hat_pair ‖ p_hat^2)`` on a grid.

  2. **Effect on the §4.5 results.** Re-run ``pipeline.decompose_session`` under
     ``closepair_density ∈ {'squared','direct'}`` and pool per-cell ``1-α``
     (naive / full / central) and the Fano factor over the fig2 good-cell mask,
     reproducing the §4.5 table under each close-pair-density estimate.

Self-contained: the only input is ``cache/aligned_sessions.pkl`` (built by
``data_loading.py``). Does not touch ``realdata_results.pkl``,
``VisionCore/covariance.py``, or any GPU cache.

Run from this folder:  uv run python compute_closepair_density.py [--recompute]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import dill

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_loading import load_cache                                  # noqa: E402
from pipeline import (                                               # noqa: E402
    decompose_session, _extract_windows_numpy, DT as PIPE_DT,
)
from estimators import (                                             # noqa: E402
    _geometric_median, _density_fn, _rms_traj_close_pairs,
)
from legacy.covariance import extract_valid_segments                 # noqa: E402

CACHE = THIS_DIR / "cache" / "closepair_density.pkl"
SCHEMA_VERSION = 1

# §4.5 estimator window (identical to generate_realdata.py)
T_HIST_MS = 92.0          # 11 history bins at 120 Hz; +1 count bin = 12-bin traj
T_COUNT_BINS = 1
MIN_SEG_LEN = 36
TARGETS = ("naive", "full", "central")

# fig2 inclusion criteria (legacy.compute_fig2_data)
MIN_RATE_HZ = 2.0
MIN_PSTH_R2 = 0.05
MIN_VAR = 0.0


# ---------------------------------------------------------------------------
# (1) density diagnostics: p_hat_pair (direct) vs p_hat^2 (squared marginal)
# ---------------------------------------------------------------------------

def _kl_pair_vs_squared(phat, phat_pair, rho, grid_n=80, pad=0.15):
    """KL(p_pair ‖ p^2) on a grid, both densities self-normalized over the grid.

    p^2 is taken as the (renormalized) squared representative-point KDE -- the
    quantity the 'squared' weights implicitly use as the close-pair density.
    """
    lo = np.percentile(rho, 0.5, axis=0) - pad
    hi = np.percentile(rho, 99.5, axis=0) + pad
    gx = np.linspace(lo[0], hi[0], grid_n)
    gy = np.linspace(lo[1], hi[1], grid_n)
    XX, YY = np.meshgrid(gx, gy)
    G = np.column_stack([XX.ravel(), YY.ravel()])
    cell = (gx[1] - gx[0]) * (gy[1] - gy[0])

    p1 = np.clip(phat(G), 1e-300, None)            # p_hat
    p_sq = p1 ** 2                                  # ∝ p_hat^2
    p_sq = p_sq / (p_sq.sum() * cell)
    p_pair = np.clip(phat_pair(G), 1e-300, None)
    p_pair = p_pair / (p_pair.sum() * cell)

    mask = p_pair > 1e-12
    kl = float(np.sum(p_pair[mask] * np.log(p_pair[mask] / p_sq[mask]) * cell))
    return kl


def density_diagnostics(rec, threshold=0.05):
    """Per-session close-pair-density comparison at the §4.5 window."""
    robs = np.nan_to_num(np.asarray(rec["robs"], float), nan=0.0)
    eyepos = np.nan_to_num(np.asarray(rec["eyepos"], float), nan=0.0)
    valid_mask = np.asarray(rec["valid_mask"], bool)

    segments = extract_valid_segments(valid_mask, min_len_bins=MIN_SEG_LEN)
    t_hist = max(int(T_HIST_MS / (PIPE_DT * 1000)), T_COUNT_BINS)
    _counts, trajectories, T_idx = _extract_windows_numpy(
        robs, eyepos, segments, T_COUNT_BINS, t_hist
    )
    if trajectories is None or len(trajectories) < 100:
        return None

    rho = _geometric_median(trajectories)                       # (N, 2)
    gi, gj, _tp, _mid = _rms_traj_close_pairs(trajectories, T_idx, threshold)
    if len(gi) < 50:
        return None
    rho_mid = 0.5 * (rho[gi] + rho[gj])                         # (P, 2)

    phat = _density_fn(rho, "kde")
    phat_pair = _density_fn(rho_mid, "kde")

    var_rho = float(np.trace(np.cov(rho.T)))
    var_mid = float(np.trace(np.cov(rho_mid.T)))
    var_ratio = var_mid / var_rho if var_rho > 0 else np.nan
    kl = _kl_pair_vs_squared(phat, phat_pair, rho)

    return dict(session=rec["session"], subject=rec["subject"],
                n_rho=int(len(rho)), n_pairs=int(len(gi)),
                var_rho=var_rho, var_mid=var_mid, var_ratio=var_ratio,
                kl_pair_vs_squared=kl)


# ---------------------------------------------------------------------------
# (2) per-session §4.5 decomposition under each close-pair-density estimate
# ---------------------------------------------------------------------------

def run_session(rec, closepair_density):
    """Per-cell 1-α (naive/full/central) and Fano (naive/full) at the §4.5
    window, under the given close-pair-density estimate."""
    C = rec["robs"].shape[2]
    nan = np.full(C, np.nan)
    rate_hz = np.asarray(rec["rate_hz"], float)
    psth_r2 = np.asarray(rec["psth_r2"], float)
    incl = (np.isfinite(rate_hz) & (rate_hz > MIN_RATE_HZ)
            & np.isfinite(psth_r2) & (psth_r2 > MIN_PSTH_R2))

    r = decompose_session(rec, windows_bins=(T_COUNT_BINS,),
                          t_hist_ms=T_HIST_MS, n_shuffles=0,
                          closepair_density=closepair_density)
    if not r["windows"]:
        return dict(naive=nan, full=nan, central=nan, fano_naive=nan,
                    fano_full=nan, good=incl, subj=np.array([rec["subject"]] * C))

    w = r["windows"][0]
    Ctotal = w["Ctotal"]
    good = incl & (np.diag(Ctotal) > MIN_VAR)
    oma = {t: w["targets"][t]["one_minus_alpha"] for t in TARGETS}

    fano = {}
    for t in ("naive", "full"):
        b = w["targets"][t]
        CnoiseC = 0.5 * ((Ctotal - b["Crate"]) + (Ctotal - b["Crate"]).T)
        with np.errstate(divide="ignore", invalid="ignore"):
            fano[t] = np.where(b["Erate"] > 0,
                               np.diag(CnoiseC) / b["Erate"], np.nan)

    return dict(naive=oma["naive"], full=oma["full"], central=oma["central"],
                fano_naive=fano["naive"], fano_full=fano["full"],
                good=good, subj=np.array([rec["subject"]] * C))


def compute():
    sessions = load_cache()
    diag = []
    keys = ("naive", "full", "central", "fano_naive", "fano_full", "good", "subj")
    pooled = {cd: {k: [] for k in keys} for cd in ("squared", "direct")}
    n_sess = 0
    for rec in sessions:
        d = density_diagnostics(rec)
        if d is not None:
            diag.append(d)
        res = {cd: run_session(rec, cd) for cd in ("squared", "direct")}
        for cd in ("squared", "direct"):
            for k in keys:
                pooled[cd][k].append(res[cd][k])
        n_sess += 1
        vr = d["var_ratio"] if d is not None else np.nan
        g = res["squared"]["good"]
        print(f"  {rec['session']:22s} good={int(g.sum()):3d}/{len(g):3d} "
              f"var_ratio={vr:.3f}  "
              f"full sq={np.nanmedian(res['squared']['full'][g]):.3f} "
              f"dir={np.nanmedian(res['direct']['full'][g]):.3f}  "
              f"cent sq={np.nanmedian(res['squared']['central'][g]):.3f} "
              f"dir={np.nanmedian(res['direct']['central'][g]):.3f}")

    out = {"schema_version": SCHEMA_VERSION, "n_sessions": n_sess,
           "diagnostics": diag}
    for cd in ("squared", "direct"):
        out[cd] = {k: np.concatenate(pooled[cd][k]) for k in keys}
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(res):
    diag = res["diagnostics"]
    vr = np.array([d["var_ratio"] for d in diag])
    kl = np.array([d["kl_pair_vs_squared"] for d in diag])
    print(f"\n=== close-pair density: p_hat_pair (direct) vs p_hat^2 "
          f"({len(diag)} sessions) ===")
    print(f"  var ratio  tr cov(rho_mid)/tr cov(rho)  (ideal Gaussian p^2 = 0.5): "
          f"median {np.median(vr):.3f}  range [{vr.min():.3f}, {vr.max():.3f}]")
    print(f"  KL(p_pair ‖ p^2):  median {np.median(kl):.4f}  "
          f"max {kl.max():.4f}")

    g = res["squared"]["good"]
    assert np.array_equal(g, res["direct"]["good"])
    subj = res["squared"]["subj"][g]
    print(f"\n=== §4.5 table under each close-pair density "
          f"(n={int(g.sum())} good cells, {len(np.unique(subj))} monkeys, "
          f"{res['n_sessions']} sessions) ===")
    print(f"  {'quantity':28s} {'squared (p^2)':>15s} {'direct (p_pair)':>15s} "
          f"{'Δ median':>10s}")
    rows = [
        ("median 1-α naive", "naive", "naive"),
        ("median 1-α full (p)", "full", "full"),
        ("median 1-α central (p^2)", "central", "central"),
    ]
    for label, ks, kd in rows:
        msq = np.nanmedian(res["squared"][ks][g])
        mdir = np.nanmedian(res["direct"][kd][g])
        print(f"  {label:28s} {msq:>15.3f} {mdir:>15.3f} {mdir - msq:>+10.3f}")

    for label, key in [("median Fano naive", "fano_naive"),
                       ("median Fano full (p)", "fano_full")]:
        msq = np.nanmedian(res["squared"][key][g])
        mdir = np.nanmedian(res["direct"][key][g])
        print(f"  {label:28s} {msq:>15.3f} {mdir:>15.3f} {mdir - msq:>+10.3f}")

    gap_sq = np.abs(res["squared"]["full"][g] - res["squared"]["central"][g])
    gap_dir = np.abs(res["direct"]["full"][g] - res["direct"]["central"][g])
    print(f"  {'median |full-central| gap':28s} {np.nanmedian(gap_sq):>15.3f} "
          f"{np.nanmedian(gap_dir):>15.3f} "
          f"{np.nanmedian(gap_dir) - np.nanmedian(gap_sq):>+10.3f}")

    # per-cell shifts (paired, same cells)
    for label, key in [("full", "full"), ("central", "central")]:
        d = res["direct"][key][g] - res["squared"][key][g]
        ok = np.isfinite(d)
        print(f"  per-cell {label:8s} direct-squared 1-α: "
              f"median {np.nanmedian(d[ok]):+.3f}  "
              f"|·| median {np.nanmedian(np.abs(d[ok])):.3f}  "
              f"p90 {np.nanpercentile(np.abs(d[ok]), 90):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    if CACHE.exists() and not args.recompute:
        with open(CACHE, "rb") as f:
            res = dill.load(f)
        if res.get("schema_version") != SCHEMA_VERSION:
            print(f"cache schema {res.get('schema_version')} != {SCHEMA_VERSION}; "
                  "recomputing")
            res = compute()
            with open(CACHE, "wb") as f:
                dill.dump(res, f)
    else:
        res = compute()
        with open(CACHE, "wb") as f:
            dill.dump(res, f)
    report(res)


if __name__ == "__main__":
    main()
