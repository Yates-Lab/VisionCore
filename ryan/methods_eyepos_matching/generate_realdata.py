r"""Real-data quantification of the eye-position-matching correction (cache-only).

Uses the fig4 cache (real V1 spikes + model-twin rates on the REAL fixational eye
trajectories, trial-aligned by stimulus phase). No GPU, no model inference.

  * 1-alpha and Fano are computed on the real SPIKES (robs), per cell, with each
    cell's own validity mask (dfs != 0 & eye-finite) -- this reproduces fig2's
    per-cell 1-alpha at the median. We report the naive estimate, the two matched
    targets ('full' -> p, 'central' -> p^2), and the full-vs-central GAP (a direct
    non-homogeneity measure; ~0 only for a homogeneous stimulus).
  * Noise correlation is illustrated on the deterministic model rates (rhat): with
    no Poisson noise the true stimulus-independent covariance is ~0, so any nonzero
    naive Ctotal-Crate is the distribution-mismatch leak; the matched estimator
    removes it.

Run from this folder:  uv run python generate_realdata.py [--recompute]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import dill
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fig4"))
from _fig4_data import load_fig4_data, CCMAX_THRESHOLD          # noqa: E402
from _style import configure, save, C_FULL, C_CLOSE, C_TRUTH    # noqa: E402

CACHE = Path(__file__).resolve().parent / "realdata_results.pkl"
THR = 0.05
MIN_TPP = 10


# ---------------------------------------------------------------------------
# Per-cell 1-alpha / Fano on real spikes (close pairs enumerated once / session)
# ---------------------------------------------------------------------------

def _percell_session(robs, eye, vm, dfs, thr=THR, weight_clip=1e6):
    """Per-cell naive/full/central 1-alpha and Fano for one session.

    Close pairs (same phase, |de|<thr) are enumerated once; each cell uses only the
    pairs and samples where it is valid (dfs!=0). Returns dict of (C,) arrays.
    """
    n_tr, n_ph, C = robs.shape
    S = np.nan_to_num(robs, nan=0.0)
    Dfs = (dfs != 0)
    # flatten to valid-eye samples
    samp = np.where(vm)
    si_tr, si_ph = samp
    Sf = S[si_tr, si_ph, :]                # (N, C)
    Ef = eye[si_tr, si_ph, :]              # (N, 2)
    Df = Dfs[si_tr, si_ph, :]              # (N, C) per-cell validity
    Tf = si_ph
    N = len(Tf)
    kde = gaussian_kde(Ef.T)
    p_samp = np.clip(kde(Ef.T), 1e-12, None)

    # enumerate close pairs once (per phase)
    pi, pj = [], []
    for t in np.unique(Tf):
        ix = np.where(Tf == t)[0]
        if len(ix) < 2:
            continue
        a, b = np.triu_indices(len(ix), k=1)
        d = np.linalg.norm(Ef[ix[a]] - Ef[ix[b]], axis=1)
        c = d < thr
        pi.append(ix[a[c]]); pj.append(ix[b[c]])
    pi = np.concatenate(pi); pj = np.concatenate(pj)
    mid = 0.5 * (Ef[pi] + Ef[pj])
    pm = np.clip(kde(mid.T), 1e-12, None)
    pw_full = np.clip(1.0 / pm, None, weight_clip * np.median(1.0 / pm))

    out = {k: np.full(C, np.nan) for k in
           ("naive", "full", "central", "fano_naive", "fano_full", "Erate")}
    SiSj = Sf[pi] * Sf[pj]                  # (P, C) per-cell pair products
    Vp = Df[pi] & Df[pj]                    # (P, C) pair valid for cell
    for c in range(C):
        vc = Df[:, c]
        if vc.sum() < 2 * MIN_TPP:
            continue
        vpc = Vp[:, c]
        if vpc.sum() < MIN_TPP:
            continue
        sc = Sf[vc, c]
        # phase pair-count weighting for PSTH split-half (per cell, c-valid)
        oma, fano, erate = {}, {}, {}
        for tgt in ("naive", "full", "central"):
            sw = p_samp[vc] if tgt == "central" else np.ones(vc.sum())
            er = np.average(sc, weights=sw)
            ctot = _wvar(sc, sw, er)
            pw = pw_full[vpc] if tgt == "full" else np.ones(vpc.sum())
            mm = np.average(SiSj[vpc, c], weights=pw)
            crate = mm - er ** 2
            cpsth = _psth_diag(sc, Tf[vc], sw)
            oma[tgt] = 1.0 - np.clip(cpsth / crate, 0, 1) if crate > 0 else np.nan
            fano[tgt] = (ctot - crate) / er if er > 0 else np.nan
            erate[tgt] = er
        out["naive"][c] = oma["naive"]; out["full"][c] = oma["full"]
        out["central"][c] = oma["central"]
        out["fano_naive"][c] = fano["naive"]; out["fano_full"][c] = fano["full"]
        out["Erate"][c] = erate["full"]
    return out


def _wvar(x, w, mu):
    """Reliability-weights unbiased variance (reduces to ddof=1 for uniform w)."""
    wn = w / w.sum()
    denom = 1.0 - np.sum(wn ** 2)
    return np.sum(wn * (x - mu) ** 2) / denom if denom > 0 else np.nan


def _psth_diag(s, t, sw, n_boot=20, seed=0):
    """Split-half PSTH variance (one cell) with sample weights sw, pair-count phases."""
    rng = np.random.default_rng(seed)
    phases = [u for u in np.unique(t) if (t == u).sum() >= MIN_TPP]
    if len(phases) < 2:
        return np.nan
    idx = {u: np.where(t == u)[0] for u in phases}
    nt = np.array([len(idx[u]) for u in phases], float)
    wph = nt * (nt - 1) / 2
    wph /= wph.sum()
    acc = 0.0
    for _ in range(n_boot):
        A, B = [], []
        for u in phases:
            ix = rng.permutation(idx[u]); m = len(ix) // 2
            wa, wb = sw[ix[:m]], sw[ix[m:]]
            A.append(np.average(s[ix[:m]], weights=wa))
            B.append(np.average(s[ix[m:]], weights=wb))
        A, B = np.array(A), np.array(B)
        acc += np.sum(wph * (A - np.average(A, weights=wph)) *
                      (B - np.average(B, weights=wph)))
    return acc / n_boot


def compute():
    data = load_fig4_data()
    pooled = {k: [] for k in ("naive", "full", "central", "fano_naive",
                              "fano_full", "good", "subj")}
    for sr in data["session_results"]:
        ccmax = np.asarray(sr["ccmax"]); good = ccmax > CCMAX_THRESHOLD
        if good.sum() < 2:
            continue
        r = _percell_session(sr["robs_used"], sr["eyepos_used"],
                             sr["valid_mask"], sr["dfs_used"])
        for k in ("naive", "full", "central", "fano_naive", "fano_full"):
            pooled[k].append(r[k])
        pooled["good"].append(good)
        pooled["subj"].append(np.array([sr["subject"]] * len(good)))
        print(f"  {sr['session']:22s} good={good.sum():3d} "
              f"naive={np.nanmedian(r['naive'][good]):.3f} "
              f"full={np.nanmedian(r['full'][good]):.3f} "
              f"central={np.nanmedian(r['central'][good]):.3f}")
    return {k: np.concatenate(v) for k, v in pooled.items()}


def report(res):
    g = res["good"]
    print("\n=== pooled good cells (n=%d) ===" % g.sum())
    for k in ("naive", "full", "central"):
        print(f"  1-alpha {k:8s}: median {np.nanmedian(res[k][g]):.3f}")
    shift = res["naive"][g] - res["full"][g]
    gap = np.abs(res["full"][g] - res["central"][g])
    print(f"  naive - full (Direction 1 shift): median {np.nanmedian(shift):+.3f}")
    print(f"  |full - central| gap (non-homogeneity): median {np.nanmedian(gap):.3f}")
    print(f"  Fano naive median {np.nanmedian(res['fano_naive'][g]):.3f}, "
          f"full median {np.nanmedian(res['fano_full'][g]):.3f}")


def make_figure(res):
    configure()
    g = res["good"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=C_TRUTH, lw=0.8, ls="--")
    ax.scatter(res["naive"][g], res["full"][g], s=10, color=C_FULL, alpha=0.5,
               label="Direction 1 (full, $p$)")
    ax.scatter(res["naive"][g], res["central"][g], s=10, color=C_CLOSE, alpha=0.5,
               marker="s", label="Direction 2 (central, $p^2$)")
    ax.set_xlabel(r"naive $1-\alpha$"); ax.set_ylabel(r"matched $1-\alpha$")
    ax.set_title("A  real spikes: naive vs matched"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    gap = np.abs(res["full"][g] - res["central"][g])
    ax.hist(gap[np.isfinite(gap)], bins=np.linspace(0, 0.5, 30), color="#8e44ad", alpha=0.8)
    ax.axvline(np.nanmedian(gap), color=C_TRUTH, ls="--",
               label=f"median {np.nanmedian(gap):.3f}")
    ax.set_xlabel(r"$|(1-\alpha)_{\rm full}-(1-\alpha)_{\rm central}|$")
    ax.set_ylabel("cell count")
    ax.set_title("B  non-homogeneity gap (real)"); ax.legend(fontsize=7)

    ax = axes[2]
    lim = (0.5, 2.0)
    ax.plot(lim, lim, color=C_TRUTH, lw=0.8, ls="--")
    ax.scatter(res["fano_naive"][g], res["fano_full"][g], s=10, color=C_FULL, alpha=0.5)
    ax.set_xlabel("naive Fano"); ax.set_ylabel("matched (full) Fano")
    ax.set_title("C  Fano shift (real spikes)")
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")

    fig.tight_layout()
    save(fig, "fig_realdata.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    if CACHE.exists() and not args.recompute:
        with open(CACHE, "rb") as f:
            res = dill.load(f)
    else:
        res = compute()
        with open(CACHE, "wb") as f:
            dill.dump(res, f)
    report(res)
    make_figure(res)


if __name__ == "__main__":
    main()
