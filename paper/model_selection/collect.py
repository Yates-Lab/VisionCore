"""Pool the Stage 0 arms into one table, refusing to mix protocols.

The sweep runs over days. Two failure modes make an unassisted reading of the
run directory untrustworthy, and this module exists to close both:

1. **Pooling across protocols.** A run trained under one split is not
   comparable to a run trained under another. Every manifest carries
   `protocol_hash`; `evaluate.load_run_manifest` refuses a mismatch by raising,
   and a raise here is the intended outcome, not an inconvenience to catch.
2. **Reading a delta with no scale.** Every arm but the baseline reports a
   ΔBPS, and a delta means nothing without the run-to-run spread. This module
   prints the deltas and points at `stability.py` for the floor; it deliberately
   does not print a verdict, because deciding whether a delta is resolved is
   that module's job and needs the replicates to exist.

Selection is on validation BPS (`protocol.SELECTION_METRIC` / `SELECTION_SPLIT`)
and validation BPS is recorded only in the checkpoint filename, so it is read
from there. Test-split and `fixrsvp` numbers come from `evaluation.json`; a run
that has not been evaluated reports an absent metric rather than a zero.

    uv run python paper/model_selection/collect.py
    uv run python paper/model_selection/collect.py --json runs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from evaluate import best_checkpoint, load_run_manifest  # noqa: E402
from launch import CHECK_VAL_EVERY, CKPT_ROOT  # noqa: E402
from protocol import PROTOCOL_HASH  # noqa: E402


# The arm every other arm is differenced against. `_b_defaults` in launch.py is
# this configuration, so F1a/F1b/F1c share it and form the replicate group that
# gives the deltas their scale.
#
# Stage 0's baseline was E1a. It is deliberately *not* the default any more:
# Stage 0b changed both the batching and the architecture, so differencing a
# Stage 0b arm against E1a would compare across two changes at once. Completed
# Stage 0 runs are still poolable on their own with `--baseline E1a`, and
# `config_signature` keeps the two families in separate replicate groups
# regardless, since `model_config` and `homogeneous` are both in CONFIG_KEYS.
BASELINE_RUN = "F1a"

# The spec fields that make two runs the same experiment. Seed is excluded on
# purpose: runs differing only in seed are replicates, not distinct arms, and
# detecting them by signature rather than by a hard-coded name list means a new
# replicate is picked up without editing this file.
CONFIG_KEYS = (
    "config", "model_config", "width", "batch_size", "lr", "core_lr_scale",
    "wd", "homogeneous", "effective_batch", "accumulate", "max_epochs",
)


# The independently settable knobs, for reporting *what* differs between two
# runs. `accumulate` is excluded because it is derived from effective_batch and
# would double-count a single change; `max_epochs` stands in for the sample
# budget, which it is derived from.
KNOB_KEYS = tuple(k for k in CONFIG_KEYS if k != "accumulate")


def spec_diff(baseline_spec, spec):
    """Which knobs differ between two runs. Empty means same configuration.

    A delta is only a measurement of one thing when one thing changed. E1a
    differs from the Stage 0b baseline in width, architecture *and* batching,
    so its ΔBPS measures their sum and is not a verdict on any of them --
    reporting it beside a single-knob arm's delta invites exactly that misread.
    """
    return [k for k in KNOB_KEYS if baseline_spec.get(k) != spec.get(k)]


def config_signature(spec):
    """The identity of an arm's *configuration*, ignoring seed.

    Two runs with the same signature are replicates of one another. This is how
    the baseline replicate group is found, rather than by naming E1a/E2b/E3b:
    the fact that those three are the same configuration is a consequence of
    `launch.py`'s defaults, and a consequence is a thing to detect, not to
    restate in a second place where it can fall out of date.
    """
    return tuple((k, spec.get(k)) for k in CONFIG_KEYS)


def val_bps_from_checkpoint(path):
    """Validation BPS as recorded in a checkpoint filename, or None.

    `ModelCheckpoint` writes `epoch=NN-val_bps_overall=X.XXXX.ckpt`. Training
    logs the metric to wandb but nothing local, so the filename is the only
    on-disk record of the selection metric.
    """
    stem = Path(path).stem
    if "val_bps_overall=" not in stem:
        return None
    try:
        return float(stem.split("val_bps_overall=")[-1])
    except ValueError:
        return None


def epoch_from_checkpoint(path):
    """The epoch index in a checkpoint filename, or None."""
    stem = Path(path).stem
    if "epoch=" not in stem:
        return None
    try:
        return int(stem.split("epoch=")[1].split("-")[0])
    except (IndexError, ValueError):
        return None


def run_status(last_epoch, max_epochs):
    """A coarse completion flag. Never gates a number, only labels one.

    Lightning validates every `CHECK_VAL_EVERY` epochs and checkpoints only on
    validation, so the last checkpointed epoch trails the horizon by up to one
    validation interval even in a finished run. A run further behind than that
    is either still training or died; this cannot tell those apart from disk
    alone, so it says `partial` and leaves the distinction to whoever is
    watching the box.
    """
    if last_epoch is None:
        return "no ckpt"
    if last_epoch >= max_epochs - 1 - CHECK_VAL_EVERY:
        return "done"
    return "partial"


def _elapsed_hours(manifest, ckpts):
    """Wall-clock from launch to the newest checkpoint, in hours.

    Approximate by construction: it ends at the last checkpoint write, not at
    process exit, and it starts when `launch.py` stamped the manifest, which
    includes the ~19 min of dataset loading. Good enough for the throughput half
    of the E1 decision; not a benchmark.
    """
    launched = manifest.get("launched")
    if not launched or not ckpts:
        return None
    try:
        start = datetime.fromisoformat(launched)
    except ValueError:
        return None
    end = datetime.fromtimestamp(max(p.stat().st_mtime for p in ckpts))
    return (end - start).total_seconds() / 3600.0


def load_run(run_dir):
    """Everything known about one run, or raise if it is not poolable.

    Raises `FileNotFoundError` if the directory carries no manifest (it was not
    produced by the Stage 0 family) and `ValueError` if it was produced under a
    different protocol. Both are conditions the caller must see.
    """
    run_dir = Path(run_dir)
    manifest = load_run_manifest(run_dir)          # protocol gate lives here
    spec = manifest.get("spec", {})

    ckpts = sorted(run_dir.glob("*.ckpt"))
    selectable = [p for p in ckpts if p.name != "last.ckpt"]
    try:
        best = best_checkpoint(run_dir)
    except FileNotFoundError:
        best = None

    epochs = [e for e in (epoch_from_checkpoint(p) for p in selectable)
              if e is not None]
    last_epoch = max(epochs) if epochs else None

    row = {
        "run": manifest.get("run", run_dir.name),
        "dir": str(run_dir),
        "note": spec.get("note", ""),
        "seed": spec.get("seed"),
        "spec": spec,
        "signature": config_signature(spec),
        "launched": manifest.get("launched"),
        "n_checkpoints": len(selectable),
        "last_epoch": last_epoch,
        "max_epochs": spec.get("max_epochs"),
        "status": run_status(last_epoch, spec.get("max_epochs") or 0),
        "best_checkpoint": best.name if best else None,
        "val_bps": val_bps_from_checkpoint(best) if best else None,
        "hours": _elapsed_hours(manifest, ckpts),
        "evaluated": False,
        "test_bps": None,
        "ccnorm": None,
        "single_trial_r2": None,
        "fixrsvp_bps": None,
        "fixrsvp_units": None,
        "fixrsvp_sessions": None,
    }

    eval_path = run_dir / "evaluation.json"
    if eval_path.exists():
        report = json.loads(eval_path.read_text())
        # An evaluation.json is stamped with the protocol it was scored under.
        # It can postdate the manifest, so it is checked in its own right.
        if report.get("protocol_hash") != PROTOCOL_HASH:
            raise ValueError(
                f"{eval_path} was scored under protocol "
                f"{report.get('protocol_hash')!r}, not {PROTOCOL_HASH!r}. "
                f"Re-run evaluate.py for this arm.")
        row["evaluated"] = True
        row["test_bps"] = (report.get("test_split") or {}).get("bps_overall")
        fix = report.get("fixrsvp") or {}
        metrics = fix.get("metrics") or {}
        row["ccnorm"] = (metrics.get("ccnorm") or {}).get("median")
        row["single_trial_r2"] = (metrics.get("single_trial_r2") or {}).get("median")
        row["fixrsvp_bps"] = (metrics.get("bps") or {}).get("median")
        row["fixrsvp_units"] = fix.get("n_units")
        row["fixrsvp_sessions"] = fix.get("n_sessions")

    return row


def load_runs(root=None, only=None):
    """Load every poolable run under `root`.

    Returns `(rows, skipped)`. A directory without a manifest is skipped with a
    reason rather than ignored: the smoke-test directories from pipeline
    validation live alongside the arms, and silently omitting a directory is how
    a real run goes missing from a table.

    A protocol mismatch is *not* skipped. It propagates, because a mixed-protocol
    pool is the failure this module exists to prevent and a warning in a long
    stdout is not a defence against it.
    """
    root = Path(root) if root else CKPT_ROOT
    rows, skipped = [], []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if only and run_dir.name not in only:
            continue
        try:
            rows.append(load_run(run_dir))
        except FileNotFoundError as exc:
            # First sentence only; the rest restates how runs are stamped. Split
            # on ". " rather than "." so the ".json" in the path survives.
            skipped.append((run_dir.name, str(exc).split(". ")[0]))
    return rows, skipped


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------
def baseline_group(rows, baseline_run=BASELINE_RUN):
    """The rows sharing the baseline's configuration signature.

    These are the replicates. Empty if the baseline itself has not run, in
    which case there is no reference and no deltas can be formed.
    """
    match = next((r for r in rows if r["run"] == baseline_run), None)
    if match is None:
        return []
    return [r for r in rows if r["signature"] == match["signature"]]


def reference_values(group, metrics=("val_bps", "test_bps", "ccnorm")):
    """Mean of each metric over the replicate group, ignoring absent values.

    The reference is the group mean rather than the single baseline run: with
    replicates in hand, the mean is the better estimate of the configuration's
    performance, and differencing against one arbitrarily chosen member imports
    that member's noise into every delta.
    """
    out = {}
    for metric in metrics:
        vals = [r[metric] for r in group if r.get(metric) is not None]
        out[metric] = sum(vals) / len(vals) if vals else None
    return out


def attach_deltas(rows, baseline_run=BASELINE_RUN):
    """Add `d_<metric>` to every row, relative to the replicate-group mean."""
    group = baseline_group(rows, baseline_run)
    ref = reference_values(group)
    group_names = {r["run"] for r in group}
    for row in rows:
        row["is_baseline_replicate"] = row["run"] in group_names
        for metric, base in ref.items():
            value = row.get(metric)
            row[f"d_{metric}"] = (
                None if base is None or value is None else value - base)
    return ref, group


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def _fmt(value, spec="{:.4f}", dash="  --  "):
    return dash if value is None else spec.format(value)


def format_table(rows, ref, group):
    lines = []
    lines.append(f"{'run':<6}{'status':>8}{'val_bps':>9}{'Δval':>8}"
                 f"{'test_bps':>10}{'Δtest':>8}{'ccnorm':>8}{'Δcc':>8}"
                 f"{'r2':>8}{'hours':>7}  note")
    for row in sorted(rows, key=lambda r: r["run"]):
        mark = "*" if row.get("is_baseline_replicate") else " "
        lines.append(
            f"{row['run']:<5}{mark}{row['status']:>8}"
            f"{_fmt(row['val_bps']):>9}{_fmt(row.get('d_val_bps'), '{:+.4f}'):>8}"
            f"{_fmt(row['test_bps']):>10}{_fmt(row.get('d_test_bps'), '{:+.4f}'):>8}"
            f"{_fmt(row['ccnorm'], '{:.3f}'):>8}"
            f"{_fmt(row.get('d_ccnorm'), '{:+.3f}'):>8}"
            f"{_fmt(row['single_trial_r2'], '{:.4f}'):>8}"
            f"{_fmt(row['hours'], '{:.1f}'):>7}  {row['note']}")

    lines.append("")
    if group:
        names = ", ".join(sorted(r["run"] for r in group))
        lines.append(f"* baseline replicate group ({names}); deltas are against "
                     f"its mean")
        lines.append(f"  reference: val_bps {_fmt(ref.get('val_bps'))}  "
                     f"test_bps {_fmt(ref.get('test_bps'))}  "
                     f"ccnorm {_fmt(ref.get('ccnorm'), '{:.3f}')}")
    else:
        lines.append(f"! no baseline ({BASELINE_RUN}) in this pool -- no deltas "
                     f"could be formed")
    lines.append("")
    lines.append("A delta is not readable without the run-to-run spread: "
                 "uv run python paper/model_selection/stability.py")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="Checkpoint root to scan")
    ap.add_argument("--runs", nargs="+", default=None, help="Limit to these runs")
    ap.add_argument("--json", default=None, help="Also write the pool here")
    ap.add_argument("--baseline", default=BASELINE_RUN)
    args = ap.parse_args()

    rows, skipped = load_runs(args.root, only=args.runs)
    if not rows:
        raise SystemExit(f"No Stage 0 runs found under "
                         f"{Path(args.root) if args.root else CKPT_ROOT}")

    ref, group = attach_deltas(rows, args.baseline)

    print(f"protocol {PROTOCOL_HASH}   "
          f"{len(rows)} runs   {sum(r['evaluated'] for r in rows)} evaluated\n")
    print(format_table(rows, ref, group))
    for name, why in skipped:
        print(f"  skipped {name}: {why}")

    if args.json:
        payload = {
            "protocol_hash": PROTOCOL_HASH,
            "baseline": args.baseline,
            "baseline_replicates": sorted(r["run"] for r in group),
            "reference": ref,
            "runs": [{k: v for k, v in r.items() if k != "signature"}
                     for r in rows],
            "skipped": [{"dir": n, "reason": w} for n, w in skipped],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
