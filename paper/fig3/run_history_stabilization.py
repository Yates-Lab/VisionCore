#!/usr/bin/env python3
"""Replay the selected Figure-3 twin with history- or trial-level stabilization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relocate_repository_path(value, source_root):
    path = Path(value)
    if not path.is_absolute():
        return source_root / path
    for anchor in ("outputs", "paper", "scripts"):
        if anchor in path.parts:
            return source_root.joinpath(*path.parts[path.parts.index(anchor):])
    return path


def resolve_replay_inputs(
    selection,
    *,
    source_root,
    empirical_cache_dir=None,
    dataset_config_path=None,
):
    """Resolve selected immutable inputs while allowing verified local substitutes."""
    source_root = Path(source_root).expanduser().resolve()
    bundle = source_root / selection["bundle"]
    final_path = bundle / "FINAL_MANIFEST.json"
    if digest(final_path) != selection["final_manifest_sha256"]:
        raise ValueError("Selected analysis completion manifest changed")
    final = json.loads(final_path.read_text())
    if final.get("status") != "complete" or final.get("checkpoint_sha256") != selection["checkpoint_sha256"]:
        raise ValueError("Selected analysis is incomplete or uses a different checkpoint")

    manifest = json.loads((bundle / "figure3/run_manifest.json").read_text())
    environment = dict(manifest["environment"])
    for key, value in list(environment.items()):
        if isinstance(value, str) and Path(value).is_absolute():
            environment[key] = str(_relocate_repository_path(value, source_root))

    if dataset_config_path is None:
        recorded = Path(manifest["environment"]["FIG3_DATASET_CONFIGS"])
        local_candidate = ROOT.joinpath(*recorded.parts[recorded.parts.index("paper"):])
        dataset_config_path = local_candidate if local_candidate.exists() else environment["FIG3_DATASET_CONFIGS"]
    dataset_config_path = Path(dataset_config_path).expanduser().resolve()
    if digest(dataset_config_path) != manifest["dataset_config_sha256"]:
        raise ValueError("Dataset config differs from the selected Figure-3 run")
    environment["FIG3_DATASET_CONFIGS"] = str(dataset_config_path)

    if empirical_cache_dir is not None:
        empirical_cache_dir = Path(empirical_cache_dir).expanduser().resolve()
        overrides = {
            "FIG3_COVDECOMP_CACHE_PATH": "covdecomp_empirical.pkl",
            "FIG3_COVDECOMP_DERIVED_CACHE_PATH": "covdecomp_derived.pkl",
            "COVDECOMP_ALIGNED_CACHE_PATH": "covdecomp_aligned_sessions.pkl",
        }
        for key, filename in overrides.items():
            path = empirical_cache_dir / filename
            expected_key = {
                "FIG3_COVDECOMP_CACHE_PATH": "covdecomp_cache_sha256",
                "FIG3_COVDECOMP_DERIVED_CACHE_PATH": "covdecomp_derived_cache_sha256",
                "COVDECOMP_ALIGNED_CACHE_PATH": "covdecomp_aligned_cache_sha256",
            }[key]
            if expected_key in manifest and digest(path) != manifest[expected_key]:
                raise ValueError(f"Local empirical substitute differs: {path}")
            environment[key] = str(path)

    checkpoint = Path(environment["FIG3_MODEL_CHECKPOINT"])
    if (
        manifest.get("checkpoint_sha256") != selection["checkpoint_sha256"]
        or digest(checkpoint) != selection["checkpoint_sha256"]
    ):
        raise ValueError("Checkpoint file differs from the selected model")
    input_sha256 = {}
    for key in (
        "FIG3_MODEL_CHECKPOINT",
        "FIG3_DATASET_CONFIGS",
        "FIG3_REFERENCE_CACHE",
        "FIG3_CACHE_PATH",
        "FIG3_ABLATION_CACHE_PATH",
        "FIG3_COVDECOMP_CACHE_PATH",
        "FIG3_COVDECOMP_DERIVED_CACHE_PATH",
        "COVDECOMP_ALIGNED_CACHE_PATH",
    ):
        path = Path(environment.get(key, ""))
        if path.is_file():
            input_sha256[key] = digest(path)
    provenance = {
        "source_root": str(source_root),
        "selected_bundle": str(bundle),
        "final_manifest_sha256": digest(final_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "dataset_config": str(dataset_config_path),
        "dataset_config_sha256": digest(dataset_config_path),
        "empirical_cache_dir": str(Path(empirical_cache_dir).resolve()) if empirical_cache_dir is not None else None,
        "resolved_environment": environment,
        "input_sha256": input_sha256,
    }
    return manifest, environment, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+")
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--reference",
        choices=("history_endpoint", "trial_centroid"),
        default="history_endpoint",
    )
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--empirical-cache-dir", type=Path)
    parser.add_argument("--merge-shards", nargs="+", type=Path,
                        help="Combine completed, disjoint session shards without inference")
    args = parser.parse_args()
    selection = json.loads((ROOT / "manuscript/analysis/selected_model_bundle.json").read_text())
    source_root = args.source_root or os.environ.get("VISIONCORE_MANUSCRIPT_SOURCE_ROOT", ROOT)
    empirical_cache_dir = args.empirical_cache_dir or os.environ.get("VISIONCORE_MANUSCRIPT_EMPIRICAL_CACHE_DIR")
    manifest, environment, input_provenance = resolve_replay_inputs(
        selection,
        source_root=source_root,
        empirical_cache_dir=empirical_cache_dir,
        dataset_config_path=args.dataset_config,
    )
    os.environ.update(environment)
    os.environ.update(FIG3_GPU=args.gpu, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_name = {
        "history_endpoint": "history_stabilized.pkl",
        "trial_centroid": "trial_stabilized.pkl",
    }[args.reference]
    cache = args.out_dir.resolve() / cache_name
    if cache.exists():
        raise FileExistsError(f"Completed cache already exists: {cache}")
    from _fig3_ablation_data import _run_inference, _write_inference_cache_atomic, _inference_cache_payload
    if args.merge_shards:
        import dill
        results = []
        for shard in args.merge_shards:
            source = shard / cache_name
            shard_manifest = json.loads((shard / "run_manifest.json").read_text())
            if (
                shard_manifest["selection"] != selection
                or shard_manifest.get("stabilization_reference") != args.reference
                or digest(source) != shard_manifest["local_cache_sha256"]
            ):
                raise ValueError(f"Shard provenance mismatch: {source}")
            payload = dill.load(source.open("rb"))
            if not payload.get("complete") or payload["checkpoint_path"] != environment["FIG3_MODEL_CHECKPOINT"]:
                raise ValueError(f"Incomplete or mismatched shard: {source}")
            if any(r.get("stabilization_reference") != args.reference for r in payload["results"]):
                raise ValueError(f"Wrong stabilization reference: {source}")
            results.extend(payload["results"])
        expected = dill.load(open(environment["FIG3_ABLATION_CACHE_PATH"], "rb"))["results"]
        lookup = {row["session"]: row for row in results}
        if len(lookup) != len(results) or set(lookup) != {row["session"] for row in expected}:
            raise ValueError("Shards must cover the exact production session set without duplicates")
        results = [lookup[row["session"]] for row in expected]
        _write_inference_cache_atomic(cache, _inference_cache_payload(results, complete=True))
    else:
        results = _run_inference(
            session_filter=set(args.sessions) if args.sessions else None,
            cache_path=cache,
            history_stabilized=args.reference == "history_endpoint",
            stabilization_reference=args.reference,
        )
    report = {
        "selection": selection,
        "stabilization_reference": args.reference,
        "global_cache": environment["FIG3_ABLATION_CACHE_PATH"],
        "local_cache": str(cache),
        "local_cache_sha256": digest(cache),
        "input_provenance": input_provenance,
        "conditions": {
            "intact": "full model",
            "zeroed": "behavior zeroed",
            "stabilized": (
                "gaze frozen at each prediction's latest input sample"
                if args.reference == "history_endpoint"
                else "gaze frozen at each source trial's centroid"
            ),
        },
        "sessions": {
            r["session"]: r.get("stabilization_render_audit", r.get("history_render_audit"))
            for r in results
        },
        "trial_boundary_audit": {
            "sessions": {r["session"]: r.get("trial_boundary_audit") for r in results},
            "totals": {
                key: sum(r.get("trial_boundary_audit", {}).get(key, 0) for r in results)
                for key in (
                    "candidate_trial_time_bins",
                    "crossing_trial_time_bins",
                    "canonical_supported_trial_time_bins",
                    "canonical_supported_crossing_bins",
                )
            },
        },
    }
    if args.merge_shards:
        report["merged_shards"] = {
            str(shard): digest(shard / cache_name) for shard in args.merge_shards
        }
    (args.out_dir / f"{Path(cache_name).stem}_run_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    # Keep the historical name when only one control is present in a directory.
    (args.out_dir / "run_manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
