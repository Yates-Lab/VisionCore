#!/usr/bin/env python3
"""Analyze corrected controlled scaling from its saved NPZ output."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import CONTROLLED_DIR, LEGACY_MATRIX_DIR


EPS = 1e-8
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260810


def main() -> int:
    with np.load(CONTROLLED_DIR / "corrected_controlled_scaling_response.npz") as archive:
        ssi = np.asarray(archive["ssi"], dtype=np.float64)
        expected = np.asarray(archive["expected_spikes"], dtype=np.float64)
        banks = archive["bank_names"].astype(str)
        scales = np.asarray(archive["scale_factors"], dtype=float)
    unit = pd.read_csv(LEGACY_MATRIX_DIR / "unit_feature_table.csv").sort_values("unit_index")
    sf = pd.to_numeric(unit["sf_split_metric"], errors="coerce").to_numpy(dtype=float)
    groups = {"low_sf_lt0p5": np.flatnonzero(sf < 0.5), "high_sf_ge0p5": np.flatnonzero(sf >= 0.5)}
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for bank_index, bank in enumerate(banks):
        # Scale zero is the matched within-bank no-scored-motion reference.
        for group, units in groups.items():
            for scale_index, scale in enumerate(scales):
                num = ssi[bank_index, :, :, scale_index][:, :, units] * expected[bank_index, :, :, scale_index][:, :, units]
                den = expected[bank_index, :, :, scale_index][:, :, units]
                base_num = ssi[bank_index, :, :, 0][:, :, units] * expected[bank_index, :, :, 0][:, :, units]
                base_den = expected[bank_index, :, :, 0][:, :, units]
                point_rate = float(np.sum(num) / max(np.sum(den), EPS))
                base_rate = float(np.sum(base_num) / max(np.sum(base_den), EPS))
                point = 100.0 * (point_rate - base_rate) / base_rate
                boot = []
                for _ in range(N_BOOTSTRAP):
                    ii = rng.integers(0, num.shape[0], num.shape[0])
                    tt = rng.integers(0, num.shape[1], num.shape[1])
                    uu = rng.integers(0, num.shape[2], num.shape[2])
                    ix = np.ix_(ii, tt, uu)
                    m = np.sum(num[ix]) / max(np.sum(den[ix]), EPS)
                    b = np.sum(base_num[ix]) / max(np.sum(base_den[ix]), EPS)
                    boot.append(100.0 * (m - b) / b)
                rows.append(
                    {
                        "bank": bank,
                        "sf_group": group,
                        "trajectory_amplitude_x": scale,
                        "ssi_percent_vs_bank_scale0": point,
                        "ci95_low_units_images_trajectories_boot": float(np.percentile(boot, 2.5)),
                        "ci95_high_units_images_trajectories_boot": float(np.percentile(boot, 97.5)),
                        "n_images": num.shape[0],
                        "n_trajectories": num.shape[1],
                        "n_units": num.shape[2],
                        "n_bootstrap": N_BOOTSTRAP,
                        "bootstrap_seed_family": BOOTSTRAP_SEED,
                    }
                )
    pd.DataFrame(rows).to_csv(CONTROLLED_DIR / "corrected_controlled_scaling_curves.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
