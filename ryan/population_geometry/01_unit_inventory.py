"""Step 1.1 — Unit inventory for the in-silico foveal population.

Builds a per-unit table across all cached fig4-checkpoint sessions joining:
  - isolation quality  : contamination % (min rejectable RPV contamination),
                         from qc/refractory/refractory.npz
  - twin fidelity      : ccnorm, ccmax, rho, ve_model, from the fig4 cache
                         (outputs/cache/fig3_digitaltwin.pkl)
  - firing rate        : from qc/refractory/refractory.npz

No GPU / model load required — reads the fig4 inference cache and the per-session
QC files. Produces distribution plots and a survivor-count grid so selection
thresholds (isolation + fidelity + rate) can be chosen from the data before we
freeze the population.

Index alignment (verified in recon):
  cache `neuron_mask[k]`  -> readout-unit index j in [0, n_units)
  session-yaml `cids[j]`  -> positional row into refractory arrays (145 sorted clusters)
so contamination[j] (== load_qc_data(sess, cids)['contamination'][j]) and
firing_rates[cids][j] line up with cache fidelity metrics at neuron_mask[k].
"""
import sys
from pathlib import Path

import numpy as np
import dill
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR, FIGURES_DIR
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

from DataYatesV1.utils.io import get_session
from eval.eval_stack_utils import load_qc_data

CACHE_PATH = CACHE_DIR / "fig3_digitaltwin.pkl"
SESSIONS_DIR = VISIONCORE_ROOT / "experiments" / "dataset_configs" / "sessions"
OUT_DIR = FIGURES_DIR / "population_geometry"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_session_cids(session_name):
    """Authoritative readout cids for a session (positional rows into refractory)."""
    with open(SESSIONS_DIR / f"{session_name}.yaml") as f:
        return np.asarray(yaml.safe_load(f)["cids"], dtype=int)


def build_inventory():
    with open(CACHE_PATH, "rb") as f:
        session_results = dill.load(f)
    print(f"Loaded fig4 cache: {len(session_results)} sessions")

    rows = []
    for sr in session_results:
        name = sr["session"]
        subject = sr["subject"]
        neuron_mask = np.asarray(sr["neuron_mask"], dtype=int)  # -> readout idx
        cids = load_session_cids(name)
        assert neuron_mask.max() < len(cids), (
            f"{name}: neuron_mask max {neuron_mask.max()} >= n_units {len(cids)}")

        subj, date = name.split("_", 1)
        sess = get_session(subj, date)
        qc = load_qc_data(sess, cids)                    # aligned to readout order
        contamination = np.asarray(qc["contamination"])  # (n_units,)
        truncation = np.asarray(qc["truncation"])         # (n_units,)
        ref = np.load(sess.sess_dir / "qc" / "refractory" / "refractory.npz")
        firing_rates = ref["firing_rates"][cids]          # (n_units,)

        for k, j in enumerate(neuron_mask):
            rows.append(dict(
                session=name, subject=subject,
                readout_idx=int(j), cluster_pos=int(cids[j]),
                contam_pct=float(contamination[j]),
                missing_pct=float(truncation[j]),
                firing_rate=float(firing_rates[j]),
                ccnorm=float(sr["ccnorm"][k]),
                ccmax=float(sr["ccmax"][k]),
                rho=float(sr["rhos"][k]),
                ve_model=float(sr["ve_model"][k]),
            ))
    df = pd.DataFrame(rows)
    print(f"Total units in cache: {len(df)} "
          f"({(df.subject=='Allen').sum()} Allen, {(df.subject=='Logan').sum()} Logan)")
    return df


def summarize(df):
    finite = df.dropna(subset=["contam_pct", "ccnorm", "firing_rate"])
    print(f"\nUnits with finite (contam, ccnorm, rate): {len(finite)}")
    for col in ["contam_pct", "ccnorm", "ccmax", "firing_rate", "missing_pct"]:
        v = df[col].to_numpy()
        v = v[np.isfinite(v)]
        qs = np.percentile(v, [5, 25, 50, 75, 95])
        print(f"  {col:12s} n={len(v):4d}  "
              f"p5={qs[0]:.2f} p25={qs[1]:.2f} med={qs[2]:.2f} "
              f"p75={qs[3]:.2f} p95={qs[4]:.2f}")

    print("\nSurvivor counts (units / sessions) by (contam_pct <, ccnorm >), rate>1Hz:")
    contam_grid = [2, 5, 10, 20]
    cc_grid = [0.4, 0.5, 0.6, 0.7]
    hdr = "  contam< | " + " | ".join(f"cc>{c:.1f}" for c in cc_grid)
    print(hdr)
    for ct in contam_grid:
        cells = []
        for cc in cc_grid:
            m = ((df.contam_pct < ct) & (df.ccnorm > cc) & (df.firing_rate > 1.0))
            sub = df[m]
            cells.append(f"{len(sub):4d}/{sub.session.nunique():2d}")
        print(f"  {ct:6d}  | " + " | ".join(cells))


def plot(df):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

    ax = axes[0, 0]
    ax.hist(df.contam_pct.clip(upper=100), bins=40, color="steelblue")
    ax.axvline(5, color="k", ls="--", lw=1)
    ax.set_xlabel("contamination % (min rejectable RPV)")
    ax.set_ylabel("units"); ax.set_title("isolation quality")

    ax = axes[0, 1]
    ax.hist(df.ccnorm.dropna(), bins=40, color="indianred")
    ax.set_xlabel("ccnorm (twin fidelity)"); ax.set_ylabel("units")
    ax.set_title("twin fidelity")

    ax = axes[0, 2]
    ax.hist(df.ccmax.dropna(), bins=40, color="seagreen")
    ax.set_xlabel("ccmax (reliability / noise ceiling)"); ax.set_ylabel("units")
    ax.set_title("response reliability")

    ax = axes[1, 0]
    ax.hist(np.log10(df.firing_rate.clip(lower=1e-2)), bins=40, color="slategray")
    ax.axvline(0, color="k", ls="--", lw=1)  # 1 Hz
    ax.set_xlabel("log10 firing rate (Hz)"); ax.set_ylabel("units")
    ax.set_title("firing rate")

    ax = axes[1, 1]
    for subj, c in [("Allen", "tab:blue"), ("Logan", "tab:green")]:
        d = df[df.subject == subj]
        ax.scatter(d.contam_pct.clip(upper=60), d.ccnorm, s=8, alpha=0.4,
                   color=c, label=subj)
    ax.axvline(5, color="k", ls="--", lw=1); ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("contamination % (clipped 60)"); ax.set_ylabel("ccnorm")
    ax.set_title("isolation vs fidelity"); ax.legend(fontsize=8)

    ax = axes[1, 2]
    for subj, c in [("Allen", "tab:blue"), ("Logan", "tab:green")]:
        d = df[df.subject == subj]
        ax.scatter(d.ccmax, d.ccnorm, s=8, alpha=0.4, color=c, label=subj)
    ax.plot([0, 1], [0, 1], color="k", lw=0.8)
    ax.set_xlabel("ccmax"); ax.set_ylabel("ccnorm")
    ax.set_title("fidelity vs reliability"); ax.legend(fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "unit_inventory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure -> {out}")


if __name__ == "__main__":
    df = build_inventory()
    df.to_csv(OUT_DIR / "unit_inventory.csv", index=False)
    summarize(df)
    plot(df)
