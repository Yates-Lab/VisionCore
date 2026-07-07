"""High-level unit/session counts for the paper, printed to the console.

Source of truth is the covariance-decomposition pipeline (Figures 2/3) — the
actual variability analysis — not the Fig 1 fixRSVP display config. We reuse
``load_empirical_data`` so every count here is guaranteed to match the figures.

Three nested unit tiers are reported per subject and overall, each with the
total, the mean +/- SD across sessions, and the range across sessions:

    sorted    spike-sorted units fed into the pipeline (meta n_neurons_total)
    aligned   units surviving fixRSVP trial alignment (meta n_neurons_used)
    analyzed  units passing the analysis inclusion filter
              (rate > MIN_RATE_HZ and split-half PSTH R^2 > MIN_PSTH_R2)

The "analyzed" tier is the population the variability analysis is performed on
and is the number to quote for "units used in the analysis". The per-window
panel-B distribution counts (analyzed units whose 1-alpha lands in [0, 1]) are
printed separately for reconciling figure captions.

Usage:
    uv run python paper/fig1/recording_unit_counts.py [--refresh]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from VisionCore.paths import VISIONCORE_ROOT

_COVDECOMP = str(VISIONCORE_ROOT / "paper" / "covariance_decomposition")
if _COVDECOMP not in sys.path:
    sys.path.insert(0, _COVDECOMP)

from derive import load_empirical_data, MIN_RATE_HZ, MIN_PSTH_R2  # noqa: E402

SUBJECT_ORDER = ["Allen", "Logan"]
SUBJECT_LABEL = {"Allen": "Monkey A (Allen)", "Logan": "Monkey L (Logan)"}


def _session_counts(session_results):
    """Per-session (subject, sorted, aligned, analyzed) unit counts."""
    rows = []
    for sr in session_results:
        rate = np.asarray(sr["rate_hz"])
        r2 = np.asarray(sr["psth_r2"])
        analyzed = int(
            (
                np.isfinite(rate) & (rate > MIN_RATE_HZ)
                & np.isfinite(r2) & (r2 > MIN_PSTH_R2)
            ).sum()
        )
        rows.append({
            "session": sr["session"],
            "subject": sr["subject"],
            "sorted": int(sr["meta"]["n_neurons_total"]),
            "aligned": int(sr["meta"]["n_neurons_used"]),
            "analyzed": analyzed,
        })
    return rows


def _summarize(values):
    """total, n_sessions, mean, SD (sample), min, max for a list of counts."""
    v = np.asarray(values, dtype=float)
    return {
        "n_sessions": int(v.size),
        "total": int(v.sum()),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        "min": int(v.min()),
        "max": int(v.max()),
    }


def _print_row(label, s):
    mean_sd = f"{s['mean']:.1f}+/-{s['sd']:.1f}"
    rng = f"{s['min']}-{s['max']}"
    print(f"  {label:<18}{s['n_sessions']:>9}{s['total']:>8}"
          f"{mean_sd:>14}{rng:>12}")


def _print_tier(name, rows, tier):
    print(f"\n{name} units per session")
    print(f"  {'subject':<18}{'sessions':>9}{'total':>8}"
          f"{'mean+/-SD':>14}{'range':>12}")
    for subj in SUBJECT_ORDER:
        vals = [r[tier] for r in rows if r["subject"] == subj]
        if not vals:
            continue
        _print_row(SUBJECT_LABEL[subj], _summarize(vals))
    _print_row("Both subjects", _summarize([r[tier] for r in rows]))


def _print_per_window(bundle):
    """Analyzed n per subject at each counting window (panel-B distribution)."""
    print("\nAnalyzed units in the 1-alpha distribution, per counting window")
    print("  (analyzed units with 1-alpha in [0, 1]; matches Fig. 2B 'n=')")
    print(f"  {'window (ms)':<14}{'Monkey A':>10}{'Monkey L':>10}{'total':>8}")
    for w_idx, w in enumerate(bundle["WINDOWS_MS"]):
        subj = np.asarray(bundle["subject_per_neuron_by_window"][w_idx])
        n_a = int((subj == "Allen").sum())
        n_l = int((subj == "Logan").sum())
        print(f"  {w:<14.2f}{n_a:>10}{n_l:>10}{n_a + n_l:>8}")


def main(refresh=False):
    bundle = load_empirical_data(refresh=refresh)
    rows = _session_counts(bundle["session_results"])

    n_sessions = len(rows)
    n_a = sum(r["subject"] == "Allen" for r in rows)
    n_l = sum(r["subject"] == "Logan" for r in rows)

    print("=" * 60)
    print("Recording / unit counts for the paper")
    print("Source: covariance-decomposition pipeline (Figures 2/3)")
    print("=" * 60)
    print(f"\nSessions: {n_sessions} total "
          f"(Monkey A: {n_a}, Monkey L: {n_l}); 2 subjects")
    print(f"Inclusion filter (analyzed): rate > {MIN_RATE_HZ} Hz "
          f"and split-half PSTH R^2 > {MIN_PSTH_R2}")

    _print_tier("Sorted (recorded)", rows, "sorted")
    _print_tier("Aligned (fixRSVP)", rows, "aligned")
    _print_tier("Analyzed (inclusion-filtered)", rows, "analyzed")

    _print_per_window(bundle)
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh", action="store_true",
                   help="Rebuild the derived bundle before counting.")
    args = p.parse_args()
    main(refresh=args.refresh)
