"""Shared paths for the orientation-corrected Figure 4 rebuild."""
from __future__ import annotations

import hashlib
from pathlib import Path

from VisionCore.paths import FIGURES_DIR, STATS_DIR

JAKE = Path("/home/jake/repos/VisionCore")
BUNDLE = JAKE / "outputs/no_phase_readout_comparison_20260910/rank1"
SOURCE = BUNDLE / "figure4"
PRODUCTION_SHARDS = [SOURCE / f"matrix_spectral_replay/shard_{i:02d}/causal_chain_shard.npz" for i in range(2)]
CORRECTED_REPLAY = STATS_DIR / "fig4_direction/corrected_orientation_replay/corrected_orientation_replay.npz"
PREDICTORS = ("total_dynamic_power", "joint_signed_rate_drive", "joint_passband_power", "tf_marginal_power",
              "sf_orientation_marginal_power", "separable_passband_power")

OUT = STATS_DIR / "fig4_orientation_fix"
FIG_OUT = FIGURES_DIR / "fig4_orientation_fix"
CORRECTED = OUT / "bundle/figure4"          # mirrors SOURCE for every rebuilt artifact


def corrected_shards() -> list[Path]:
    return [CORRECTED / f"matrix_spectral_replay/shard_{i:02d}/causal_chain_shard.npz" for i in range(2)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()
