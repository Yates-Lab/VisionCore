"""Rerun Jake's passband-engagement comparison (jake/passband_comparison) on orientation-corrected engagement.

The released analysis inputs are reused except `primary_engagement` / `movie_engagement`, which are recomputed from the
given spectral shards exactly as jake/passband_comparison/data.py does. Two local deviations, both reproducibility-only:
  * StratifiedGroupKFold assignments differ under this environment's scikit-learn, so the released fold assignments
    are reused (the seeds map onto the released repeats one-to-one; secondary eye-trial folds equal primary repeat 0).
  * The released design's source hashes are keyed to Jake's repository root, so `source_sha256` is emptied and the
    shard hashes are recorded under `orientation_correction` instead.
Run with --variant production as a control: it must reproduce the released summary.

Usage:
    .venv/bin/python declan/fig4_orientation_fix/passband_comparison.py --variant corrected
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JAKE, OUT, PRODUCTION_SHARDS, corrected_shards, sha256  # noqa: E402

RELEASED = JAKE / "outputs/passband_comparison_20260914"


def engagement(shards):
    scenes, power = [], []
    for path in shards:
        with np.load(path) as z:
            scenes.extend(z["image_indices"].tolist())
            power.append(z["joint_passband_power"])
    power = np.concatenate(power, axis=0)[np.argsort(scenes)].astype(np.float64)
    movie = power[:, :, 1] - power[:, :, 0]
    return np.median(movie, axis=0), movie


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("production", "corrected"), required=True)
    args = ap.parse_args()
    shards = PRODUCTION_SHARDS if args.variant == "production" else corrected_shards()
    out = OUT / f"passband_comparison_{args.variant}"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("traces.csv", "images.csv", "units.csv", "input_audit.json"):
        shutil.copy2(RELEASED / name, out / name)
    inputs = dict(np.load(RELEASED / "analysis_inputs.npz"))
    ep, movie = engagement(shards)
    if args.variant == "production" and not (np.array_equal(ep, inputs["primary_engagement"])
                                             and np.array_equal(movie, inputs["movie_engagement"])):
        raise RuntimeError("production shards do not reproduce the released engagement inputs")
    inputs.update(primary_engagement=ep, movie_engagement=movie)
    np.savez_compressed(out / "analysis_inputs.npz", **inputs)
    design = json.loads((RELEASED / "design.json").read_text())
    design["released_source_sha256"] = design.pop("source_sha256")
    design["source_sha256"] = {}
    design["orientation_correction"] = {
        "variant": args.variant, "spectral_shards": {str(p): sha256(p) for p in shards},
        "released_analysis": str(RELEASED),
        "fold_assignments": "released StratifiedGroupKFold assignments reused (scikit-learn version differs)"}
    (out / "design.json").write_text(json.dumps(design, indent=2) + "\n")

    released_folds = np.load(RELEASED / "primary_predictions_lambda_0.01.npz")["fold_assignments"]
    from jake.passband_comparison.data import SEED
    import jake.passband_comparison.estimator_diagnostics as diag
    import jake.passband_comparison.finalize as fin
    import jake.passband_comparison.normalized_overlap as norm
    import jake.passband_comparison.run as run

    def trial_folds(_table, seed, n_splits=5):
        return released_folds[seed - SEED].copy()
    run.trial_folds = norm.trial_folds = trial_folds

    for module, argv in ((run, ["--out-dir", str(out)]), (diag, ["--out-dir", str(out)]),
                         (fin, ["--out-dir", str(out)]), (norm, ["--input-dir", str(out)])):
        sys.argv = [module.__file__, *argv]
        module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
