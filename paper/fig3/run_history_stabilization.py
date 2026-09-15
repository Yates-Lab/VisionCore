#!/usr/bin/env python3
"""Replay the selected Figure-3 twin with history-local gaze stabilization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--merge-shards", nargs="+", type=Path,
                        help="Combine completed, disjoint session shards without inference")
    args = parser.parse_args()
    selection = json.loads((ROOT / "manuscript/analysis/selected_model_bundle.json").read_text())
    bundle = ROOT / selection["bundle"]
    manifest = json.loads((bundle / "figure3/run_manifest.json").read_text())
    checkpoint = Path(manifest["environment"]["FIG3_MODEL_CHECKPOINT"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
        raise ValueError("Checkpoint file differs from the selected model")
    os.environ.update(manifest["environment"])
    os.environ.update(FIG3_GPU=args.gpu, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.out_dir.resolve() / "history_stabilized.pkl"
    if cache.exists():
        raise FileExistsError(f"Completed cache already exists: {cache}")
    from _fig3_ablation_data import _run_inference, _write_inference_cache_atomic, _inference_cache_payload
    if args.merge_shards:
        import dill
        results = []
        for shard in args.merge_shards:
            source = shard / "history_stabilized.pkl"
            shard_manifest = json.loads((shard / "run_manifest.json").read_text())
            if shard_manifest["selection"] != selection or hashlib.sha256(source.read_bytes()).hexdigest() != shard_manifest["local_cache_sha256"]:
                raise ValueError(f"Shard provenance mismatch: {source}")
            payload = dill.load(source.open("rb"))
            if not payload.get("complete") or payload["checkpoint_path"] != manifest["environment"]["FIG3_MODEL_CHECKPOINT"]:
                raise ValueError(f"Incomplete or mismatched shard: {source}")
            if any(r.get("stabilization_reference") != "history_endpoint" for r in payload["results"]):
                raise ValueError(f"Wrong stabilization reference: {source}")
            results.extend(payload["results"])
        expected = dill.load(open(manifest["environment"]["FIG3_ABLATION_CACHE_PATH"], "rb"))["results"]
        lookup = {row["session"]: row for row in results}
        if len(lookup) != len(results) or set(lookup) != {row["session"] for row in expected}:
            raise ValueError("Shards must cover the exact production session set without duplicates")
        results = [lookup[row["session"]] for row in expected]
        _write_inference_cache_atomic(cache, _inference_cache_payload(results, complete=True))
    else:
        results = _run_inference(session_filter=set(args.sessions) if args.sessions else None,
                                 cache_path=cache, history_stabilized=True)
    report = {"selection": selection, "global_cache": manifest["environment"]["FIG3_ABLATION_CACHE_PATH"],
              "local_cache": str(cache), "local_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
              "conditions": {"intact": "full model", "zeroed": "behavior zeroed",
                             "stabilized": "gaze frozen at each prediction's latest input sample"},
              "sessions": {r["session"]: r["history_render_audit"] for r in results}}
    if args.merge_shards:
        report["merged_shards"] = {str(shard): hashlib.sha256((shard / "history_stabilized.pkl").read_bytes()).hexdigest()
                                   for shard in args.merge_shards}
    (args.out_dir / "run_manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
