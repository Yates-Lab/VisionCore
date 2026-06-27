"""Survey every unit in the lead example session for a good Panel A/B demo cell.

Renders the v3 "unaccounted-for variability" layout (Panel A: eye traces + that
unit's spike rates; Panel B: the decomposition curve) for every unit, one page
per unit in a multipage PDF, sorted by a composite quality score that favors:

  * high firing rate,
  * high FEM variability *relative to total* (Cfem / Ctotal),
  * low internal variability *relative to total* (Sigma_int / Ctotal).

The score is the mean of three percentile ranks (rate high, Cfem/Ctotal high,
Sigma_int/Ctotal low), so it is robust to the different scales/units. The eye
traces and matched/divergent windows are shared across units (eye position does
not depend on the unit); only the Panel A spike rates and the Panel B curve are
swapped per unit.

    uv run paper/fig2/survey_lead_cells.py
    uv run paper/fig2/survey_lead_cells.py --refresh   # recompute the decomp cache
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpecFromSubplotSpec

from _panel_common import FIG_DIR
from generate_panel_example import (
    SESSION, _FIG2_RC, _decompose_all_units,
    plot_eye_rate_example, plot_unaccounted_variance_panel,
)


def _rank_hi(x):
    """Percentile rank in [0, 1]; larger value -> larger rank."""
    x = np.asarray(x, float)
    return np.argsort(np.argsort(x)) / max(len(x) - 1, 1)


def _score_units(allu, min_rate_hz=2.0):
    """Composite quality score per unit and the component fractions.

    Returns (score, fem_frac, int_frac, valid). Invalid units get score -inf.
    """
    Ctotal = allu["Ctotal"]
    Crate = allu["Crate"]
    Cpsth = allu["Cpsth"]
    sigma_int = allu["sigma_int"]
    rate = allu["rate_hz"]

    cfem = np.clip(Crate - Cpsth, 0.0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        fem_frac = np.where(Ctotal > 1e-6, cfem / Ctotal, np.nan)
        int_frac = np.where(Ctotal > 1e-6,
                            np.clip(sigma_int, 0.0, None) / Ctotal, np.nan)

    valid = (np.isfinite(fem_frac) & np.isfinite(int_frac)
             & (rate > min_rate_hz) & (Crate > 1e-4) & (Ctotal > 1e-4))

    score = np.full(len(Ctotal), -np.inf)
    idx = np.where(valid)[0]
    if idx.size:
        s = (_rank_hi(rate[idx]) + _rank_hi(fem_frac[idx])
             + (1.0 - _rank_hi(int_frac[idx]))) / 3.0
        score[idx] = s
    return score, fem_frac, int_frac, valid


def _render_unit_page(pdf, allu, j, rank, score, fem_frac, int_frac):
    nm = np.asarray(allu["neuron_mask"])
    unit_id = int(nm[j])
    decomp = {
        "bin_centers": allu["bin_centers"],
        "cum_crate": allu["cum_crate"][:, j],
        "Ctotal": float(allu["Ctotal"][j]),
        "Crate": float(allu["Crate"][j]),
        "Cpsth": float(allu["Cpsth"][j]),
        "sigma_int": float(allu["sigma_int"][j]),
    }

    fig = plt.figure(figsize=(6.3, 3.3))
    outer = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 1.0], wspace=0.32,
        left=0.10, right=0.975, top=0.80, bottom=0.15,
    )
    gs_a = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0, 0], height_ratios=[1.0, 1.0], hspace=0.18,
    )
    ax_eye = fig.add_subplot(gs_a[0, 0])
    ax_spk = fig.add_subplot(gs_a[1, 0], sharex=ax_eye)
    plot_eye_rate_example(ax_eye, ax_spk, unit_orig=unit_id)
    ax_eye.text(-0.20, 1.12, "A", transform=ax_eye.transAxes,
                fontweight="bold", fontsize=10, va="top", ha="left")

    ax_b = fig.add_subplot(outer[0, 1])
    plot_unaccounted_variance_panel(ax_b, decomp=decomp, caption=False)
    ax_b.xaxis.label.set_size(8)
    ax_b.yaxis.label.set_size(8)
    ax_b.tick_params(labelsize=7)
    ax_b.text(-0.20, 1.12, "B", transform=ax_b.transAxes,
              fontweight="bold", fontsize=10, va="top", ha="left")

    fig.suptitle(
        f"rank {rank}   unit {unit_id}   score={score[j]:.3f}\n"
        f"rate={allu['rate_hz'][j]:.1f} Hz   "
        f"Cfem/Ctot={fem_frac[j]:.2f}   Σint/Ctot={int_frac[j]:.2f}   "
        f"Ctot={allu['Ctotal'][j]:.3f} spk²",
        fontsize=8, y=0.99,
    )
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Survey lead-cell candidates (v3).")
    p.add_argument("--refresh", action="store_true",
                   help="Recompute the all-units decomposition cache.")
    args, _ = p.parse_known_args()

    allu = _decompose_all_units(refresh=args.refresh)
    score, fem_frac, int_frac, valid = _score_units(allu)
    nm = np.asarray(allu["neuron_mask"])

    order = np.argsort(-score)                      # best first, invalid (-inf) last
    order = [int(j) for j in order if np.isfinite(score[j])]

    print(f"Session {SESSION}: {len(nm)} units, {len(order)} pass the quality "
          f"filter (rate>2 Hz, finite decomposition).")
    print("Top 5 candidates (unit, score, rate Hz, Cfem/Ctot, Σint/Ctot):")
    for r, j in enumerate(order[:5], 1):
        print(f"  {r}. unit {int(nm[j])}  score={score[j]:.3f}  "
              f"rate={allu['rate_hz'][j]:.1f}  fem/tot={fem_frac[j]:.2f}  "
              f"int/tot={int_frac[j]:.2f}")

    out = FIG_DIR / "lead_cell_survey_v3.pdf"
    with PdfPages(out) as pdf, plt.rc_context(_FIG2_RC):
        for rank, j in enumerate(order, 1):
            _render_unit_page(pdf, allu, j, rank, score, fem_frac, int_frac)
    print(f"\nSaved {out} ({len(order)} pages, sorted best -> worst).")


if __name__ == "__main__":
    main()
