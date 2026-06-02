"""Fig. 7 (writeup §7.2): equivalence between the legacy snapshot and the
methods pipeline at ``target='naive'``.

Three-panel scatter on the canonical window (4 bins ≈ 16.7 ms at 120 Hz):

  A. legacy diag(Crate)  vs  methods diag(Crate)
  B. legacy diag(Cpsth)  vs  methods diag(Cpsth)
  C. legacy 1-α          vs  methods 1-α

Per-panel annotations: Pearson r, slope-through-origin, and the median
absolute difference. Pass criteria (printed to stdout, written into the
caption text):

  Pearson r ≥ 0.99 on Crate / Cpsth diag,
  |Δ pop-median (1-α)| ≤ 0.002.

If criteria fail, the figure still saves but the script exits non-zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _style import configure, save, C_FULL, C_TRUTH                # noqa: E402

METHODS_DECOMP = THIS_DIR / "cache" / "methods_decomposition.pkl"
LEGACY_DECOMP = THIS_DIR / "cache" / "legacy_decomposition.pkl"

# Canonical window for the equivalence diagnostic. Index 1 of WINDOW_BINS =
# [1, 2, 3, 6] -> t_count=2 (≈16.7 ms). Matches legacy SUBSPACE_WINDOW_IDX.
W_IDX = 1

# Inclusion filter constants -- same as metrics.py
MIN_RATE_HZ = 2.0
MIN_PSTH_R2 = 0.05
MIN_VAR = 0


def _load(p):
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. Run "
            "`uv run python compute_methods_data.py --both` first."
        )
    with open(p, "rb") as f:
        return dill.load(f)


def _collect_naive_per_cell(methods_pkl, legacy_pkl, w_idx=W_IDX):
    """Pair up per-cell diagonals between the two pipelines at target='naive'.

    Inclusion: methods + legacy must both have finite Erate, Crate diag,
    Cpsth diag, and pass the rate/PSTH-R^2 cuts (the same cuts used in
    metrics.py). Cells are matched by neuron index within session -- both
    pipelines see the same `aligned['robs']` columns so indices align.
    """
    leg_results = legacy_pkl["results"]
    met_results = methods_pkl["results"]
    leg_by_sess = {sr["session"]: sr for sr in leg_results}
    met_by_sess = {sr["session"]: sr for sr in met_results}
    common = sorted(set(leg_by_sess) & set(met_by_sess))

    rows = []
    for sess in common:
        lsr = leg_by_sess[sess]
        msr = met_by_sess[sess]
        if w_idx >= len(lsr["mats"]) or w_idx >= len(msr["windows"]):
            continue
        lm = lsr["mats"][w_idx]
        mw = msr["windows"][w_idx]
        if "naive" not in mw["targets"]:
            continue
        mt = mw["targets"]["naive"]

        l_total = lm["Total"]
        l_crate = lm["Intercept"]
        l_cpsth = lm["PSTH"]
        l_er = lsr["results"][w_idx]["Erates"]

        m_total = mw["Ctotal"]
        m_crate = mt["Crate"]
        m_cpsth = mt["Cpsth"]
        m_er = mt["Erate"]

        rate_hz = lsr["rate_hz"]
        psth_r2 = lsr["psth_r2"]

        n_cells = l_crate.shape[0]
        for c in range(n_cells):
            include = (
                np.isfinite(l_er[c]) and np.isfinite(m_er[c])
                and np.isfinite(rate_hz[c]) and rate_hz[c] > MIN_RATE_HZ
                and np.isfinite(psth_r2[c]) and psth_r2[c] > MIN_PSTH_R2
                and l_total[c, c] > MIN_VAR and m_total[c, c] > MIN_VAR
                and np.isfinite(l_crate[c, c]) and np.isfinite(m_crate[c, c])
                and np.isfinite(l_cpsth[c, c]) and np.isfinite(m_cpsth[c, c])
                and l_crate[c, c] > 0 and m_crate[c, c] > 0
            )
            if not include:
                continue
            rows.append(dict(
                session=sess, cell=c,
                l_crate=l_crate[c, c], m_crate=m_crate[c, c],
                l_cpsth=l_cpsth[c, c], m_cpsth=m_cpsth[c, c],
                l_oma=1 - l_cpsth[c, c] / l_crate[c, c],
                m_oma=1 - m_cpsth[c, c] / m_crate[c, c],
            ))
    if not rows:
        raise RuntimeError("No cells pass inclusion -- check the pipeline caches.")
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]
           if k not in ("session", "cell")}
    arr["session"] = np.array([r["session"] for r in rows])
    arr["cell"] = np.array([r["cell"] for r in rows])
    return arr


def _scatter_panel(ax, x, y, label_x, label_y, color, title, log=False):
    ax.scatter(x, y, s=8, color=color, alpha=0.5, edgecolors="none")
    lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
    if log:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot([lo, hi], [lo, hi], color=C_TRUTH, lw=0.8, ls="--")
    ax.set_xlabel(label_x); ax.set_ylabel(label_y); ax.set_title(title)
    ax.set_aspect("equal")

    r = np.corrcoef(x, y)[0, 1]
    slope = np.sum(x * y) / np.sum(x ** 2)
    med_abs_d = float(np.median(np.abs(y - x)))
    txt = (f"r = {r:.4f}\nslope = {slope:.3f}\n"
           f"med |Δ| = {med_abs_d:.3g}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=7, bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.7))
    return r, slope, med_abs_d


def main():
    configure()
    methods_pkl = _load(METHODS_DECOMP)
    legacy_pkl = _load(LEGACY_DECOMP)

    win_bins = methods_pkl["windows_bins"]
    w_idx = min(W_IDX, len(win_bins) - 1)
    print(f"Equivalence diagnostic at window={win_bins[w_idx]} bins")

    pairs = _collect_naive_per_cell(methods_pkl, legacy_pkl, w_idx=w_idx)
    print(f"Paired cells: {len(pairs['l_crate'])} from "
          f"{len(np.unique(pairs['session']))} sessions")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    r_c, slope_c, dmed_c = _scatter_panel(
        axes[0], pairs["l_crate"], pairs["m_crate"],
        "legacy diag(Crate)", "methods diag(Crate)",
        C_FULL, "A  Crate diagonal", log=True,
    )
    r_p, slope_p, dmed_p = _scatter_panel(
        axes[1], pairs["l_cpsth"], pairs["m_cpsth"],
        "legacy diag(Cpsth)", "methods diag(Cpsth)",
        C_FULL, "B  Cpsth diagonal", log=True,
    )
    r_a, slope_a, dmed_a = _scatter_panel(
        axes[2], pairs["l_oma"], pairs["m_oma"],
        r"legacy $1-\alpha$", r"methods $1-\alpha$",
        C_FULL, r"C  $1-\alpha$", log=False,
    )
    axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1)

    fig.suptitle(
        f"Pipeline equivalence at window={win_bins[w_idx]} bins "
        f"(target='naive', n={len(pairs['l_crate'])} cells)",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "fig_pipeline_equivalence.png")

    # Pop-median 1-α difference
    pop_med_l = float(np.median(pairs["l_oma"]))
    pop_med_m = float(np.median(pairs["m_oma"]))
    print(f"\nPop median 1-α: legacy={pop_med_l:.4f}, methods={pop_med_m:.4f}, "
          f"|Δ|={abs(pop_med_l - pop_med_m):.4f}")
    print(f"Crate r={r_c:.4f}, Cpsth r={r_p:.4f}, 1-α r={r_a:.4f}")
    print(f"med |Δ|: Crate={dmed_c:.3g}, Cpsth={dmed_p:.3g}, 1-α={dmed_a:.3g}")

    ok_crate = r_c >= 0.99
    ok_cpsth = r_p >= 0.99
    ok_alpha = abs(pop_med_l - pop_med_m) <= 0.002
    print(f"\nPass criteria:")
    print(f"  Crate Pearson r ≥ 0.99: {'PASS' if ok_crate else 'FAIL'} ({r_c:.4f})")
    print(f"  Cpsth Pearson r ≥ 0.99: {'PASS' if ok_cpsth else 'FAIL'} ({r_p:.4f})")
    print(f"  |Δ pop-median 1-α| ≤ 0.002: "
          f"{'PASS' if ok_alpha else 'FAIL'} ({abs(pop_med_l-pop_med_m):.4f})")

    if not (ok_crate and ok_cpsth and ok_alpha):
        sys.exit(1)


if __name__ == "__main__":
    main()
