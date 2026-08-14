"""Per-session census of the training data, and the data-path regression gate.

Two jobs, one expensive pass over the sessions:

1. **Gate.** Hash the exact indices `prepare_data` serves for every session and
   sub-dataset. Figures 1-4 load a checkpoint whose readouts are sized from the
   resolved `cids`, and whose numbers come from this split. Any shared-code
   change that leaves these hashes untouched cannot have moved those figures.
   Run before and after a change and diff the JSON.

2. **Census.** Record per-session sample counts, unit counts, and spike totals.
   This is the evidence base for the subject-weighting question: whether Allen
   dominates training, and through which mechanism.

Usage
-----
    uv run python paper/model_selection/data_census.py --out census_before.json
    # ... make a change ...
    uv run python paper/model_selection/data_census.py --out census_after.json
    uv run python paper/model_selection/data_census.py --diff census_before.json census_after.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
VISIONCORE_ROOT = HERE.parent.parent
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

DEFAULT_CONFIG = (
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
)


def _hash_inds(inds) -> str:
    """Stable hash of an index tensor. Order matters: the loader serves in order."""
    arr = np.asarray(inds, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def census_one(cfg, keep_pixel_norm=False):
    """Census one session. Returns a plain dict, or None if the session fails."""
    from models.data import prepare_data
    from DataYatesV1.utils.data.loading import remove_pixel_norm

    name = cfg["session"]
    if not keep_pixel_norm:
        # Matches MultiDatasetDM's uint8 path: keeps stim as uint8 in RAM.
        cfg, _ = remove_pixel_norm(cfg)

    try:
        train_dset, val_dset, cfg_out = prepare_data(cfg, strict=True)
    except Exception as e:  # noqa: BLE001 - a failed session is data, not a crash
        print(f"  [{name}] FAILED: {e}")
        return {"session": name, "error": str(e)}

    cids = cfg_out.get("cids", None)
    n_units = len(cids) if cids is not None else int(train_dset.dsets[0]["robs"].shape[1])

    per_type = []
    train_spikes = 0.0
    for i, sub in enumerate(train_dset.dsets):
        tr_inds = train_dset.dset_inds[i]
        va_inds = val_dset.dset_inds[i]
        robs = sub["robs"]
        spikes = float(robs[tr_inds].sum())
        train_spikes += spikes
        per_type.append({
            "type": sub.metadata.get("name", f"type{i}"),
            "n_train": int(len(tr_inds)),
            "n_val": int(len(va_inds)),
            "train_spikes": spikes,
            "train_hash": _hash_inds(tr_inds),
            "val_hash": _hash_inds(va_inds),
        })

    rec = {
        "session": name,
        "subject": name.split("_")[0],
        "n_units": n_units,
        "cids_hash": _hash_inds(cids) if cids is not None else None,
        "n_train": int(len(train_dset)),
        "n_val": int(len(val_dset)),
        "train_spikes": train_spikes,
        "spikes_per_unit": train_spikes / max(n_units, 1),
        "types": per_type,
    }

    del train_dset, val_dset
    gc.collect()
    return rec


def run_census(config_path, limit=None, keep_pixel_norm=False, sessions=None):
    from models.config_loader import load_dataset_configs

    cfgs = load_dataset_configs(str(config_path))
    if sessions:
        want = set(sessions)
        cfgs = [c for c in cfgs if c["session"] in want]
        missing = want - {c["session"] for c in cfgs}
        if missing:
            raise SystemExit(f"sessions not in {config_path}: {sorted(missing)}")
    if limit is not None:
        cfgs = cfgs[:limit]

    records = []
    for i, cfg in enumerate(cfgs):
        print(f"\n=== [{i+1}/{len(cfgs)}] {cfg['session']} ===")
        records.append(census_one(cfg, keep_pixel_norm=keep_pixel_norm))
    return records


def summarize(records):
    """Per-subject rollup: the numbers the balancing argument turns on."""
    ok = [r for r in records if "error" not in r]
    by_subject = {}
    for r in ok:
        s = by_subject.setdefault(r["subject"], {
            "sessions": 0, "units": 0, "train_samples": 0, "train_spikes": 0.0})
        s["sessions"] += 1
        s["units"] += r["n_units"]
        s["train_samples"] += r["n_train"]
        s["train_spikes"] += r["train_spikes"]

    tot_sess = sum(s["sessions"] for s in by_subject.values()) or 1
    tot_samp = sum(s["train_samples"] for s in by_subject.values()) or 1
    tot_unit = sum(s["units"] for s in by_subject.values()) or 1
    for s in by_subject.values():
        s["session_share"] = s["sessions"] / tot_sess
        s["sample_share"] = s["train_samples"] / tot_samp
        s["unit_share"] = s["units"] / tot_unit

    return {"n_sessions_ok": len(ok),
            "n_sessions_failed": len(records) - len(ok),
            "by_subject": by_subject}


def print_summary(summary):
    print("\n" + "=" * 72)
    print(f"{'subject':<10} {'sess':>5} {'units':>7} {'samples':>12} "
          f"{'sess%':>7} {'samp%':>7} {'unit%':>7}")
    print("-" * 72)
    for subj, s in sorted(summary["by_subject"].items()):
        print(f"{subj:<10} {s['sessions']:>5} {s['units']:>7} "
              f"{s['train_samples']:>12,} "
              f"{100*s['session_share']:>6.1f}% {100*s['sample_share']:>6.1f}% "
              f"{100*s['unit_share']:>6.1f}%")
    print("=" * 72)
    print("Cross-session batching weights each *session* present in a batch "
          "equally\n(loss is a masked mean per session, then a mean over "
          "sessions), so sess%\nis the effective per-step subject weight. "
          "Homogeneous batching draws a session\nwith p proportional to its "
          "size, making samp% the weight instead.")


def diff(before_path, after_path):
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())
    b = {r["session"]: r for r in before["records"]}
    a = {r["session"]: r for r in after["records"]}

    problems = []
    for sess in sorted(set(b) | set(a)):
        if sess not in b or sess not in a:
            problems.append(f"{sess}: present in only one census")
            continue
        rb, ra = b[sess], a[sess]
        if rb.get("cids_hash") != ra.get("cids_hash"):
            problems.append(f"{sess}: cids changed "
                            f"({rb.get('cids_hash')} -> {ra.get('cids_hash')})")
        hb = [(t["type"], t["train_hash"], t["val_hash"]) for t in rb.get("types", [])]
        ha = [(t["type"], t["train_hash"], t["val_hash"]) for t in ra.get("types", [])]
        if hb != ha:
            problems.append(f"{sess}: served split indices changed")

    if problems:
        print("DATA PATH CHANGED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"Data path unchanged across {len(a)} sessions "
          f"(cids and every train/val index set bitwise identical).")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=None, help="Write census JSON here.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only census the first N sessions.")
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="Only census these named sessions.")
    ap.add_argument("--keep-pixel-norm", action="store_true",
                    help="Do not strip pixelnorm (uses far more RAM).")
    ap.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="Compare two census files and exit.")
    args = ap.parse_args()

    if args.diff:
        raise SystemExit(diff(*args.diff))

    records = run_census(args.config, limit=args.limit,
                         keep_pixel_norm=args.keep_pixel_norm,
                         sessions=args.sessions)
    summary = summarize(records)
    print_summary(summary)

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = HERE / out
        out.write_text(json.dumps(
            {"config": str(args.config), "summary": summary, "records": records},
            indent=2))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
