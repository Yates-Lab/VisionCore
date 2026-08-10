"""How large a ΔBPS has to be before it means anything.

Every arm but the baseline reports a delta, and a delta without a scale is not
a result. `launch.py` keeps E1a, E2b and E3b as the *same* configuration at
seeds 101/102/103 precisely so that scale exists: whatever those three runs
spread over is what two identical configurations do, and any arm whose delta
sits inside that spread has not been shown to differ from the baseline.

Two things to keep in view when reading the output.

**The floor understates the true spread.** `split_inds_by_trial*` calls
`set_seeds(SPLIT_SEED)` internally, so the trials and their order are fixed
regardless of `--seed`. The replicates therefore differ in weight
initialisation and GPU nondeterminism only, not in data order, and the run-to-
run variation a full re-randomisation would produce is larger than what is
measured here. An arm inside the floor is **unresolved, not null.**

**This is a descriptive rule, not a test.** Three runs support no useful
distributional claim. The rule below compares a delta against the observed
spread of identical configurations; that is the honest reading of n = 3, and
calling it a p-value would not make it stronger.

    uv run python paper/model_selection/stability.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from collect import BASELINE_RUN, attach_deltas, load_runs  # noqa: E402
from protocol import PROTOCOL_HASH, SELECTION_METRIC, SELECTION_SPLIT  # noqa: E402


# Metrics carried through the whole sweep. `val_bps` is the selection metric and
# is the one a decision is made on; the other two describe the selected model.
METRICS = ("val_bps", "test_bps", "ccnorm")

# Minimum replicates before a floor is reported at all. Two runs give a single
# difference, which is a spread estimate in name only -- it is reported, loudly
# qualified, because waiting for the third costs six GPU-hours and a reader who
# knows n is not misled.
MIN_REPLICATES = 2


def spread(values):
    """Descriptive spread of a replicate group.

    `range` (max - min) is what the verdict uses: with three runs it is the
    directly observed answer to "how far apart do identical configurations
    land", and it needs no distributional assumption. `sd` is reported beside
    it for readers who want one, not because n = 3 justifies it.
    """
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return {"n": len(values),
                "mean": values[0] if values else None,
                "sd": None, "range": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values),
        "range": max(values) - min(values),
        "min": min(values),
        "max": max(values),
    }


def verdict(delta, floor):
    """Classify a delta against the replicate floor.

    - `resolved`   -- larger than the full spread of identical configurations.
    - `marginal`   -- larger than their sd but inside their range.
    - `unresolved` -- inside the noise. Not evidence of no effect; evidence
                      that this sweep cannot see one. The distinction matters
                      because the floor is itself an underestimate.
    """
    if delta is None or floor is None or floor.get("range") is None:
        return "n/a"
    magnitude = abs(delta)
    if magnitude > floor["range"]:
        return "resolved"
    if floor.get("sd") is not None and magnitude > floor["sd"]:
        return "marginal"
    return "unresolved"


def missing_replicates(rows, baseline_run=BASELINE_RUN):
    """Arms that would join the baseline group but have not produced a value.

    Answers "what do I still have to run before the floor is real" without the
    reader having to hold `launch.py`'s defaults in their head. Reads the run
    families directly, so a replicate added there shows up here.

    Searches both the live and the frozen family, because the baseline can sit
    in either: Stage 0b's is F1a, while a Stage 0 pool re-read with
    `--baseline E1a` needs E2b and E3b found in FROZEN_RUNS.

    Works before the baseline itself has run, by taking the signature from its
    *declared* spec rather than from a completed run. Otherwise the one moment
    the question matters most -- nothing has run yet -- is the one moment it
    could not be answered.
    """
    from launch import FROZEN_RUNS, RUNS, resolve
    from collect import config_signature

    declared = {**FROZEN_RUNS, **RUNS}

    match = next((r for r in rows if r["run"] == baseline_run), None)
    if match is not None:
        signature = match["signature"]
    elif baseline_run in declared:
        signature = config_signature(resolve(baseline_run))
    else:
        return []

    have = {r["run"] for r in rows
            if r["signature"] == signature and r["val_bps"] is not None}
    return [name for name in declared
            if name not in have
            and config_signature(resolve(name)) == signature]


def independent_knobs(base_spec, spec):
    """`spec_diff`, reduced to the knobs an experiment can set independently.

    At accumulate 1 the micro-batch fixes the effective batch, and at a fixed
    sample budget it fixes the epoch count too. `02_lr1e-3_bs128` therefore
    differs from F2a in three `CONFIG_KEYS` but in only one *decision*, and
    counting three would file the cleanest single-knob comparison in the sweep
    as confounded and refuse it a verdict.

    The reduction is `axes_for` -- the same derivation that builds the labels
    -- applied to the experiment's own arms plus the baseline being compared
    against. A knob the experiment never varies independently is not a knob
    this comparison changed.
    """
    from collect import spec_diff
    from experiments import axes_for, experiment_of

    knobs = spec_diff(base_spec, spec)
    experiment = experiment_of(spec)
    if experiment is None or len(knobs) < 2:
        return knobs

    from launch import FROZEN_RUNS, RUNS, resolve

    members = [resolve(n) for n in list(FROZEN_RUNS) + list(RUNS)]
    members = [m for m in members if experiment_of(m) == experiment]
    if not members:
        return knobs
    axes = axes_for(members + [base_spec])
    return [k for k in knobs if k in axes] or knobs


def analyse(rows, baseline_run=BASELINE_RUN):
    """Floors per metric, and a verdict for every single-knob arm.

    Arms differing from the baseline in more than one knob are separated out
    and given no verdict. Their delta is real but it measures the *sum* of the
    changes, and a floor cannot say which one moved it -- printing it in the
    same column as a single-knob arm's delta is how a confound gets read as a
    result.
    """
    from collect import spec_diff

    ref, group = attach_deltas(rows, baseline_run)
    floors = {m: spread([r[m] for r in group]) for m in METRICS}

    base = next((r for r in rows if r["run"] == baseline_run), None)
    base_spec = base["spec"] if base else {}

    arms, confounded = [], []
    for row in sorted(rows, key=lambda r: r["run"]):
        if row.get("is_baseline_replicate"):
            continue
        knobs = independent_knobs(base_spec, row["spec"]) if base else []
        single = len(knobs) == 1
        entry = {
            "run": row["run"],
            "note": row["note"],
            "knobs": knobs,
            **{m: row.get(m) for m in METRICS},
            **{f"d_{m}": row.get(f"d_{m}") for m in METRICS},
            **{f"verdict_{m}": (verdict(row.get(f"d_{m}"), floors[m])
                                if single else "confounded")
               for m in METRICS},
        }
        (arms if single else confounded).append(entry)
    return {"reference": ref, "group": group, "floors": floors,
            "arms": arms, "confounded": confounded}


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def _fmt(value, spec="{:.4f}"):
    return "  --  " if value is None else spec.format(value)


def format_report(result, rows, baseline_run=BASELINE_RUN):
    group, floors, arms = result["group"], result["floors"], result["arms"]
    lines = []

    lines.append(f"Baseline replicates (selection metric: {SELECTION_METRIC} on "
                 f"{SELECTION_SPLIT})")
    if not group:
        lines.append(f"  none -- {baseline_run} has not run")
    else:
        lines.append(f"  {'run':<6}{'seed':>6}{'val_bps':>10}{'test_bps':>10}"
                     f"{'ccnorm':>9}")
        for row in sorted(group, key=lambda r: r["run"]):
            lines.append(f"  {row['run']:<6}{str(row['seed']):>6}"
                         f"{_fmt(row['val_bps']):>10}{_fmt(row['test_bps']):>10}"
                         f"{_fmt(row['ccnorm'], '{:.3f}'):>9}")

    # protocol.py names the selection metric "bps" on split "val"; this table
    # calls that column val_bps.
    n = floors["val_bps"]["n"]
    lines.append("")
    if n < MIN_REPLICATES:
        pending = missing_replicates(rows, baseline_run)
        lines.append(f"No floor: {n} replicate(s) with a value. "
                     f"Every delta below is unreadable until at least "
                     f"{MIN_REPLICATES} exist.")
        if pending:
            lines.append(f"  still to run: {', '.join(pending)}")
    else:
        lines.append(f"{'metric':<10}{'n':>3}{'mean':>10}{'sd':>10}{'range':>10}")
        for metric in METRICS:
            f = floors[metric]
            lines.append(f"{metric:<10}{f['n']:>3}{_fmt(f['mean']):>10}"
                         f"{_fmt(f['sd']):>10}{_fmt(f['range']):>10}")
        if n == 2:
            lines.append("  n = 2: the 'spread' is one difference. Treat every "
                         "verdict as provisional until the third replicate runs.")

    lines.append("")
    lines.append(f"{'run':<6}{'Δval':>9}{'':>2}{'verdict':<12}{'Δtest':>9}"
                 f"{'':>2}{'verdict':<12}  knob / note")
    for arm in arms:
        lines.append(
            f"{arm['run']:<6}{_fmt(arm['d_val_bps'], '{:+.4f}'):>9}  "
            f"{arm['verdict_val_bps']:<12}"
            f"{_fmt(arm['d_test_bps'], '{:+.4f}'):>9}  "
            f"{arm['verdict_test_bps']:<12}  {arm['knobs'][0]}: {arm['note']}")
    if not arms:
        lines.append("  (no single-knob arms have run yet)")

    if result.get("confounded"):
        lines.append("")
        lines.append("Not comparable to this baseline -- more than one knob "
                     "differs, so the delta measures their sum:")
        for arm in result["confounded"]:
            lines.append(
                f"{arm['run']:<6}{_fmt(arm['d_val_bps'], '{:+.4f}'):>9}  "
                f"{'(' + ', '.join(arm['knobs']) + ')'}")

    lines.append("")
    lines.append("Seeds vary weight init and GPU nondeterminism only -- "
                 "`split_inds_by_trial*` re-seeds globally, so the replicates")
    lines.append("share their data order. The floor is therefore an "
                 "underestimate, and `unresolved` means this sweep cannot see a")
    lines.append("difference, not that there is none.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--baseline", default=BASELINE_RUN)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows, _ = load_runs(args.root)
    if not rows:
        raise SystemExit("No Stage 0 runs found")

    result = analyse(rows, args.baseline)
    print(f"protocol {PROTOCOL_HASH}\n")
    print(format_report(result, rows, args.baseline))

    if args.json:
        payload = {
            "protocol_hash": PROTOCOL_HASH,
            "baseline": args.baseline,
            "replicates": sorted(r["run"] for r in result["group"]),
            "floors": result["floors"],
            "arms": result["arms"],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
