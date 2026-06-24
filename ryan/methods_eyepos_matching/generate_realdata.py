r"""Real-data payoff for the eye-position-matching correction (writeup §4.5).

Fully self-contained and isolated from every other component of the project:
the ONLY input is ``cache/aligned_sessions.pkl`` -- the fig2-inclusion-criteria
aligned arrays (real fixRSVP spikes + real fixational eye trajectories, all 25
sessions), built once by ``data_loading.py``. There is no dependency on the
fig4 panel-D cache, no GPU, and no model inference.

Methodology mirrors the rest of §4 exactly:

  * the §4.4 TRAJECTORY-mode estimator, run through the same production code
    path as the §6 pipeline (``pipeline.decompose_session``): each sample's
    eye-trajectory window is reduced to its GEOMETRIC MEDIAN, close pairs are
    filtered on the whole-window RMS trajectory distance, and a single KDE
    supplies the §4.2 importance weights -- the production-setting estimator of
    §4.4, not the single-bin stand-in the old draft used;
  * the t_hist/t_count window split of §4.4: the close-pair match is on the
    ~100 ms (12-bin) trajectory window (the neuron's integration context), but
    the spike COUNT feeding Ctotal/Crate/Fano is the single ~8 ms bin at its
    end. The small count window keeps the Fano factor (a count-per-mean, NOT a
    window-stable ratio like 1-alpha) in its reported sub-Poisson regime;
  * run per session (so close pairs are never formed across sessions), on the
    REAL spikes, with each cell carrying the session's shared validity window;
  * per-cell 1-alpha and Fano pooled across sessions over the fig2 good-cell
    inclusion mask (rate_hz > 2 Hz AND split-half PSTH r^2 > 0.05 -- the exact
    cut of legacy.compute_fig2_data), so the population is the same one Figure 2
    reports.

We report, per cell, the naive estimate, the two matched targets
('full' -> p, 'central' -> p^2), the full-vs-central GAP (a direct
fixation-scale spatial-structure measure), and the Fano factor (naive vs full).

Run from this folder:  uv run python generate_realdata.py [--recompute]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import dill
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_loading import load_cache                                # noqa: E402
from pipeline import decompose_session, DT as PIPE_DT               # noqa: E402
from _style import configure, save, C_FULL, C_CLOSE, C_TRUTH       # noqa: E402

CACHE = THIS_DIR / "realdata_results.pkl"
# 4 = directly-estimated close-pair density (the estimator's new default since
#     2026-06-24; see note_closepair_density.md). decompose_session inherits
#     closepair_density='direct'.
SCHEMA_VERSION = 4

# --- estimator window (the §4.4 t_hist/t_count decoupling) ----------------
# The close-pair match is on the WHOLE eye-trajectory window (the neuron's
# integration context, §4.4), reduced to its geometric median; the spike COUNT
# -- and hence Ctotal, Crate and the Fano factor -- is over the small t_count
# window only. We use a ~100 ms trajectory (12 bins, exactly §4.4's T_WINDOW)
# with a SINGLE-bin (~8 ms) count window, the regime in which the FEM-corrected
# Fano factor is reported. (Summing the count over the whole trajectory window
# would silently change the counting timescale and inflate the Fano, which is a
# count-per-mean ratio -- not the dimensionless 1-alpha, which is window-stable.)
T_HIST_MS = 92.0         # -> 11 bins at 120 Hz; +1 count bin = 12-bin trajectory
T_COUNT_BINS = 1         # single ~8.3 ms spike-count bin
TARGETS = ("naive", "full", "central")

# --- fig2 inclusion criteria (legacy.compute_fig2_data) -------------------
MIN_RATE_HZ = 2.0
MIN_PSTH_R2 = 0.05
MIN_VAR = 0.0

C_GAP = "#8e44ad"        # purple -- non-homogeneity gap


# ---------------------------------------------------------------------------
# Per-session decomposition via the production estimator (geometric-median
# trajectory mode, uncentred Crate -- identical code path to the §7 pipeline).
# The diagonal of every matrix is the per-cell result.
# ---------------------------------------------------------------------------

def run_session(rec):
    C = rec["robs"].shape[2]
    nan = np.full(C, np.nan)
    rate_hz = np.asarray(rec["rate_hz"], float)
    psth_r2 = np.asarray(rec["psth_r2"], float)
    incl = (np.isfinite(rate_hz) & (rate_hz > MIN_RATE_HZ)
            & np.isfinite(psth_r2) & (psth_r2 > MIN_PSTH_R2))

    r = decompose_session(rec, windows_bins=(T_COUNT_BINS,),
                          t_hist_ms=T_HIST_MS, n_shuffles=0)
    if not r["windows"]:
        return dict(naive=nan, full=nan, central=nan, fano_naive=nan,
                    fano_full=nan, good=incl, subj=np.array([rec["subject"]] * C),
                    _n_pairs=0, _n_windows=0)

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

    return dict(
        naive=oma["naive"], full=oma["full"], central=oma["central"],
        fano_naive=fano["naive"], fano_full=fano["full"],
        good=good, subj=np.array([rec["subject"]] * C),
        _n_pairs=int(w["n_close_pairs"]), _n_windows=int(w["n_samples"]),
    )


def compute():
    sessions = load_cache()
    keys = ("naive", "full", "central", "fano_naive", "fano_full", "good", "subj")
    pooled = {k: [] for k in keys}
    n_sess = 0
    for rec in sessions:
        r = run_session(rec)
        for k in keys:
            pooled[k].append(r[k])
        n_sess += 1
        g = r["good"]
        print(f"  {rec['session']:22s} good={g.sum():3d}/{len(g):3d} "
              f"win={r['_n_windows']:4d} pairs={r['_n_pairs']:6d} "
              f"naive={np.nanmedian(r['naive'][g]):.3f} "
              f"full={np.nanmedian(r['full'][g]):.3f} "
              f"central={np.nanmedian(r['central'][g]):.3f}")
    out = {k: np.concatenate(v) for k, v in pooled.items()}
    out["schema_version"] = SCHEMA_VERSION
    out["n_sessions"] = n_sess
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(res):
    g = res["good"]
    subj = res["subj"][g]
    n_subj = len(np.unique(subj))
    print(f"\n=== pooled fig2-good cells "
          f"(n={int(g.sum())}, {n_subj} monkeys, {res['n_sessions']} sessions) ===")
    for k in ("naive", "full", "central"):
        v = res[k][g]
        print(f"  1-alpha {k:8s}: median {np.nanmedian(v):.3f}  "
              f"(n_finite={np.isfinite(v).sum()})")
    shift = res["naive"][g] - res["full"][g]
    gap = np.abs(res["full"][g] - res["central"][g])
    print(f"  naive - full (Direction-1 shift): median {np.nanmedian(shift):+.3f}")
    print(f"  |full - central| gap            : median {np.nanmedian(gap):.3f}  "
          f"(p90 {np.nanpercentile(gap, 90):.3f})")
    fn, ff = res["fano_naive"][g], res["fano_full"][g]
    print(f"  Fano naive median {np.nanmedian(fn):.3f} -> "
          f"full median {np.nanmedian(ff):.3f} "
          f"(median per-cell shift {np.nanmedian(ff - fn):+.3f})")


# ---------------------------------------------------------------------------
# Figure 5 -- 2 rows x 3 columns
# ---------------------------------------------------------------------------

def _scatter(ax, x, y, color, xlabel, ylabel, title, lim=(0, 1)):
    ax.plot(lim, lim, color=C_TRUTH, lw=0.8, ls="--", zorder=1)
    ok = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[ok], y[ok], s=9, color=color, alpha=0.45, lw=0, zorder=2)
    med = np.nanmedian(y[ok] - x[ok])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.text(0.04, 0.96, f"$\\Delta$median {med:+.3f}\n$n={int(ok.sum())}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            color=C_TRUTH)


def make_figure(res):
    configure()
    g = res["good"]
    naive, full, central = res["naive"][g], res["full"][g], res["central"][g]
    fano_n, fano_f = res["fano_naive"][g], res["fano_full"][g]
    gap = np.abs(full - central)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))

    # --- top row: all three pairwise combinations of naive / central / full ---
    _scatter(axes[0, 0], naive, full, C_FULL,
             r"naive $1-\alpha$", r"full ($p$) $1-\alpha$",
             "A  naive vs full (Direction 1)")
    _scatter(axes[0, 1], naive, central, C_CLOSE,
             r"naive $1-\alpha$", r"central ($p^2$) $1-\alpha$",
             "B  naive vs central (Direction 2)")
    _scatter(axes[0, 2], full, central, C_GAP,
             r"full ($p$) $1-\alpha$", r"central ($p^2$) $1-\alpha$",
             "C  full vs central")

    # --- bottom-left: the headline distribution -- full (p) 1-alpha -----------
    ax = axes[1, 0]
    vals = full[np.isfinite(full)]
    ax.hist(vals, bins=np.linspace(0, 1, 30), color=C_FULL, alpha=0.85)
    ax.axvline(np.median(vals), color=C_TRUTH, ls="--",
               label=f"median {np.median(vals):.3f}")
    ax.set_xlabel(r"full ($p$) $1-\alpha$"); ax.set_ylabel("cell count")
    ax.set_title(r"D  FEM fraction $1-\alpha$ (Direction 1)")
    ax.legend(fontsize=8)

    # --- bottom-middle: non-homogeneity gap -----------------------------------
    ax = axes[1, 1]
    gv = gap[np.isfinite(gap)]
    ax.hist(gv, bins=np.linspace(0, 0.5, 30), color=C_GAP, alpha=0.85)
    ax.axvline(np.median(gv), color=C_TRUTH, ls="--",
               label=f"median {np.median(gv):.3f}")
    ax.set_xlabel(r"$|(1-\alpha)_{\rm full}-(1-\alpha)_{\rm central}|$")
    ax.set_ylabel("cell count")
    ax.set_title("E  fixation-scale structure gap")
    ax.legend(fontsize=8)

    # --- bottom-right: Fano shift ---------------------------------------------
    ax = axes[1, 2]
    lim = (0.2, 1.8)
    ax.plot(lim, lim, color=C_TRUTH, lw=0.8, ls="--")
    ok = np.isfinite(fano_n) & np.isfinite(fano_f)
    ax.scatter(fano_n[ok], fano_f[ok], s=9, color=C_FULL, alpha=0.45, lw=0)
    ax.set_xlabel("naive Fano"); ax.set_ylabel(r"matched (full, $p$) Fano")
    ax.set_title("F  Fano shift")
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.text(0.04, 0.96,
            f"median {np.nanmedian(fano_n[ok]):.3f}$\\to${np.nanmedian(fano_f[ok]):.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7, color=C_TRUTH)

    fig.tight_layout()
    save(fig, "fig_realdata.png")


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
    make_figure(res)


if __name__ == "__main__":
    main()
