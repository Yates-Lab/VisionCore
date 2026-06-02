"""Fig. 9 (writeup §7.4): per-session wall-time, methods vs legacy snapshot.

Reads ``cache/timing.csv`` produced by ``timing.py``. Two panels:

  A. Per-session bar chart at the canonical window (t_count=2), one pair of
     bars per session, sorted by methods runtime.
  B. Mean per-session wall-time by window (1, 2, 3, 6 bins), with subject-
     averaged error bars across sessions.

Annotations: total wall-time at each window, geometric-mean speedup.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _style import configure, save, C_FULL                         # noqa: E402

TIMING_CSV = THIS_DIR / "cache" / "timing.csv"

PIPELINE_COLOR = {"methods": C_FULL, "legacy": "#888888"}
W_CANONICAL = 2


def _load():
    if not TIMING_CSV.exists():
        raise FileNotFoundError(
            f"{TIMING_CSV} does not exist. Run `uv run python timing.py` first."
        )
    rows = []
    with open(TIMING_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                session=r["session"], subject=r["subject"],
                n_cells=int(r["n_cells"]), t_count=int(r["t_count"]),
                pipeline=r["pipeline"], wall_s=float(r["wall_s"]),
            ))
    return rows


def main():
    configure()
    rows = _load()

    sessions_in_order = sorted(set(r["session"] for r in rows))
    pipelines = ("methods", "legacy")

    # Per-session bar at canonical window
    can_rows = [r for r in rows if r["t_count"] == W_CANONICAL]
    by_sess = {}
    for r in can_rows:
        by_sess.setdefault(r["session"], {})[r["pipeline"]] = r["wall_s"]
    sessions_can = sorted(by_sess, key=lambda s: by_sess[s].get("methods", 0))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    ax = axes[0]
    x = np.arange(len(sessions_can))
    bw = 0.4
    for k, p in enumerate(pipelines):
        ys = np.array([by_sess[s].get(p, np.nan) for s in sessions_can])
        ax.bar(x + (k - 0.5) * bw, ys, width=bw,
               color=PIPELINE_COLOR[p], label=p, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in sessions_can],
                       rotation=90, fontsize=6)
    ax.set_ylabel("wall time (s)")
    ax.set_title(f"A  per-session, window={W_CANONICAL} bins")
    ax.legend(loc="upper left", fontsize=8)

    # Per-window summary
    ax = axes[1]
    windows = sorted(set(r["t_count"] for r in rows))
    width = 0.35
    for k, p in enumerate(pipelines):
        means, sds, totals = [], [], []
        for w in windows:
            vals = np.array([r["wall_s"] for r in rows
                             if r["pipeline"] == p and r["t_count"] == w])
            means.append(vals.mean()); sds.append(vals.std())
            totals.append(vals.sum())
        means = np.array(means); sds = np.array(sds); totals = np.array(totals)
        xs = np.arange(len(windows)) + (k - 0.5) * width
        ax.bar(xs, means, yerr=sds, width=width,
               color=PIPELINE_COLOR[p], label=f"{p} (Σ={totals.sum():.0f}s)",
               edgecolor="none")
    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels([f"{w}" for w in windows])
    ax.set_xlabel("window (bins)")
    ax.set_ylabel("mean wall time per session (s)")
    ax.set_title("B  per-window mean ± sd")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "fig_pipeline_speed.png")

    # Print summary table -- writeup §7.4 numbers come from this.
    print("\n--- per-window wall-time summary ---")
    print(f"{'window':>6} {'methods Σ':>10} {'legacy Σ':>10} {'speedup':>8}")
    for w in windows:
        m = sum(r["wall_s"] for r in rows
                if r["pipeline"] == "methods" and r["t_count"] == w)
        l = sum(r["wall_s"] for r in rows
                if r["pipeline"] == "legacy" and r["t_count"] == w)
        print(f"{w:>6} {m:>10.1f} {l:>10.1f} {l/max(m,1e-6):>7.2f}x")
    m_total = sum(r["wall_s"] for r in rows if r["pipeline"] == "methods")
    l_total = sum(r["wall_s"] for r in rows if r["pipeline"] == "legacy")
    print(f"  TOTAL: methods={m_total:.1f}s, legacy={l_total:.1f}s, "
          f"speedup={l_total/max(m_total,1e-6):.2f}x")
    _ = sessions_in_order


if __name__ == "__main__":
    main()
