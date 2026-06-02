"""Fig. 8 (writeup §7.3): the corrections at population scale.

Three rows × three targets on the canonical window:

  Row 1: 1-α distribution (histogram, with population median annotated).
  Row 2: Fano scatter -- corrected (CnoiseC/Erate) vs uncorrected
         (CnoiseU/Erate). Slope-through-origin fit, geometric-mean ratio.
  Row 3: Per-session Δz (Fisher z(corrected) - z(uncorrected)) bar plot
         coloured by subject.

Columns: naive / full / central. The naive column reproduces the legacy
look; the full / central columns show the shift induced by the eye-
distribution-matched estimator.
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

from _style import configure, save, C_FULL, C_CLOSE, C_TRUTH       # noqa: E402

METHODS_DERIVED = THIS_DIR / "cache" / "methods_derived.pkl"

W_IDX = 1
TARGET_ORDER = ("naive", "full", "central")
TARGET_LABEL = {"naive": "naive (legacy)",
                "full": "Direction 1 (full, $p$)",
                "central": "Direction 2 (central, $p^2$)"}
TARGET_COLOR = {"naive": C_TRUTH, "full": C_FULL, "central": C_CLOSE}
SUBJECT_COLORS = {"Allen": "tab:blue", "Logan": "tab:green",
                  "Luke": "tab:orange"}


def _load():
    if not METHODS_DERIVED.exists():
        raise FileNotFoundError(
            f"{METHODS_DERIVED} does not exist. Run "
            "`uv run python compute_methods_data.py` first."
        )
    with open(METHODS_DERIVED, "rb") as f:
        return dill.load(f)


def _panel_alpha(ax, m_dict, target):
    alpha = m_dict["alpha"]
    oma = 1 - alpha
    ok = np.isfinite(oma) & (oma >= 0) & (oma <= 1)
    vals = oma[ok]
    med = float(np.median(vals))
    bins = np.linspace(0, 1, 41)
    ax.hist(vals, bins=bins, color=TARGET_COLOR[target], alpha=0.75)
    ax.axvline(med, color=C_TRUTH, lw=1.0, ls="--",
               label=f"median {med:.3f}")
    ax.set_xlim(0, 1); ax.set_xlabel(r"$1-\alpha$")
    ax.set_ylabel("cells")
    ax.set_title(TARGET_LABEL[target])
    ax.legend(loc="upper right", fontsize=7)


def _panel_fano(ax, m_dict, target):
    ff_u, ff_c = m_dict["uncorr"], m_dict["corr"]
    ok = np.isfinite(ff_u) & np.isfinite(ff_c) & (ff_u > 0) & (ff_c > 0)
    u, c = ff_u[ok], ff_c[ok]
    erate_v = m_dict["erate"][ok]
    var_u = u * erate_v; var_c = c * erate_v
    slope_u = np.sum(erate_v * var_u) / np.sum(erate_v ** 2)
    slope_c = np.sum(erate_v * var_c) / np.sum(erate_v ** 2)
    g_u = float(np.exp(np.mean(np.log(u)))) if u.size else np.nan
    g_c = float(np.exp(np.mean(np.log(c)))) if c.size else np.nan
    ax.scatter(u, c, s=6, color=TARGET_COLOR[target], alpha=0.5,
               edgecolors="none")
    lo, hi = 0.4, max(u.max(), c.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color=C_TRUTH, lw=0.7, ls="--")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Fano (uncorrected)")
    ax.set_ylabel("Fano (corrected)")
    ax.text(0.04, 0.96,
            f"geomean unc={g_u:.3f}\ngeomean cor={g_c:.3f}\n"
            f"slope unc={slope_u:.3f}\nslope cor={slope_c:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))


def _panel_nc(ax, m_dict, target):
    z_u_ds = m_dict["rho_u_meanz_by_ds"]
    z_c_ds = m_dict["rho_c_meanz_by_ds"]
    dz_ds = m_dict["rho_delta_meanz_by_ds"]
    subjects_ds = np.asarray(m_dict["subject_by_ds"])
    sessions_ds = np.array([f"ds{i}" for i in range(len(dz_ds))])
    order = np.argsort(dz_ds)
    xs = np.arange(len(dz_ds))
    colors = [SUBJECT_COLORS.get(s, "gray") for s in subjects_ds[order]]
    ax.bar(xs, dz_ds[order], color=colors, edgecolor="none")
    ax.axhline(0, color=C_TRUTH, lw=0.6)
    dz_mean = float(np.mean(dz_ds[np.isfinite(dz_ds)]))
    ax.axhline(dz_mean, color="black", lw=0.8, ls="--",
               label=f"mean Δz = {dz_mean:+.3f}")
    ax.set_xticks([])
    ax.set_xlabel("sessions (sorted by Δz)")
    ax.set_ylabel(r"$\Delta\bar z = \bar z_{\rm cor} - \bar z_{\rm unc}$")
    ax.legend(loc="upper left", fontsize=7)


def main():
    configure()
    derived = _load()
    win_ms = derived["windows_ms"]
    win_bins = derived["windows_bins"]
    w_idx = min(W_IDX, len(win_bins) - 1)
    print(f"Corrections at window={win_bins[w_idx]} bins ({win_ms[w_idx]:.1f} ms)")

    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9.5))
    for col, tgt in enumerate(TARGET_ORDER):
        m_dict = derived["metrics"][tgt][w_idx]
        _panel_alpha(axes[0, col], m_dict, tgt)
        _panel_fano(axes[1, col], m_dict, tgt)
        _panel_nc(axes[2, col], m_dict, tgt)

    for row, label in enumerate(("A  $1-\\alpha$",
                                  "B  Fano factor",
                                  r"C  noise correlation $\Delta\bar z$")):
        axes[row, 0].text(-0.32, 0.5, label, rotation=90,
                          transform=axes[row, 0].transAxes,
                          va="center", ha="center", fontsize=11,
                          fontweight="bold")
    fig.suptitle(
        f"Eye-distribution-matching corrections at population scale "
        f"(window={win_bins[w_idx]} bins ≈ {win_ms[w_idx]:.1f} ms)",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "fig_pipeline_corrections.png")

    # Brief textual summary -- the writeup quotes these numbers in §7.3.
    print("\n--- target population summaries ---")
    for tgt in TARGET_ORDER:
        m = derived["metrics"][tgt][w_idx]
        oma = 1 - m["alpha"]
        ok = np.isfinite(oma) & (oma >= 0) & (oma <= 1)
        ff_u = m["uncorr"]; ff_c = m["corr"]
        pos = np.isfinite(ff_u) & np.isfinite(ff_c) & (ff_u > 0) & (ff_c > 0)
        dz = m["rho_delta_meanz_by_ds"]
        print(f"  {tgt:7s}  median 1-α = {np.median(oma[ok]):.3f}  "
              f"med Fano cor = {np.median(ff_c[pos]):.3f}  "
              f"mean Δz (per-session) = {np.mean(dz[np.isfinite(dz)]):+.4f}")


if __name__ == "__main__":
    main()
