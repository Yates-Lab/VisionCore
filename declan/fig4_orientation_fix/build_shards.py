"""Write orientation-corrected copies of the production Figure 4 spectral-replay shards.

Each shard is the production archive with its six spectral predictors replaced by the assay-convention values from
declan/fig4_direction/corrected_orientation_replay.py, and `average_power` mirrored onto the assay orientation axis
(bins 10..170 deg are symmetric about 90, so theta -> 180 - theta is an index reversal). Responses, axes and metadata are
copied unchanged, so every downstream production script can read these shards in place of the released ones.

Usage:
    .venv/bin/python declan/fig4_orientation_fix/build_shards.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import CORRECTED_REPLAY, OUT, PRODUCTION_SHARDS, PREDICTORS, corrected_shards, sha256  # noqa: E402


def main() -> int:
    corrected = np.load(CORRECTED_REPLAY)
    image_rows = {int(i): r for r, i in enumerate(corrected["image_indices"])}
    for source, target in zip(PRODUCTION_SHARDS, corrected_shards()):
        target.parent.mkdir(parents=True, exist_ok=True)
        with np.load(source) as z:
            data = {k: z[k] for k in z.files}
        rows = [image_rows[int(i)] for i in data["image_indices"]]
        for key in ("trace_indices", "unit_indices", "motion_scales", "orientation_deg"):
            if not np.array_equal(data[key], corrected[key]):
                raise RuntimeError(f"{key} differs between production shard and corrected replay")
        for key in ("mean_rate", "expected_spikes", "map_ssi"):
            if not np.array_equal(data[key], corrected[key][rows]):
                raise RuntimeError(f"response array {key} differs from the corrected replay copy")
        for key in PREDICTORS:
            if not np.array_equal(data[key][:, :, 1], corrected[f"production_{key}"][rows]):
                raise RuntimeError(f"production {key} is not the archived production baseline")
            if np.any(data[key][:, :, 0]):
                raise RuntimeError(f"stabilized {key} is not zero")
            data[key] = corrected[key][rows].astype(data[key].dtype)
        if not np.allclose(data["orientation_deg"], 180.0 - data["orientation_deg"][::-1]):
            raise RuntimeError("orientation bins are not symmetric about 90 degrees")
        data["average_power"] = np.ascontiguousarray(data["average_power"][..., ::-1])
        np.savez_compressed(target, **data)
        summary = json.loads((source.parent / "summary.json").read_text())
        summary["archive"] = str(target)
        summary["orientation_correction"] = {
            "source_shard": str(source), "source_shard_sha256": sha256(source),
            "corrected_replay": str(CORRECTED_REPLAY), "corrected_replay_sha256": sha256(CORRECTED_REPLAY),
            "change": "spectral predictors use mode orientation from kxy=(kx,+fy), the grating-assay convention; "
                      "average_power orientation axis mirrored; responses unchanged",
        }
        (target.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {target}")
    (OUT / "bundle").mkdir(parents=True, exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
