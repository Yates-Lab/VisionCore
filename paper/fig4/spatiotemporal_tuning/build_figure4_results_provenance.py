#!/usr/bin/env python3
"""Bind Figure 4 manuscript claims to audited production artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-audit", type=Path, required=True)
    parser.add_argument("--panel-a-audit", type=Path, required=True)
    parser.add_argument("--panel-b-reduction", type=Path, required=True)
    parser.add_argument("--tuning-release-audit", type=Path, required=True)
    parser.add_argument("--tuning-contract", type=Path, required=True)
    parser.add_argument("--rucci-ensemble", type=Path, required=True)
    parser.add_argument("--passband-comparison", type=Path, required=True)
    parser.add_argument("--stage-trajectory", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def main() -> int:
    args = parse_args()
    production = load_json(args.production_audit)
    if not bool(production.get("release_ready")):
        raise ValueError("Figure 4 production audit has not passed")

    panel_a_path = args.panel_a_audit / "summary.json"
    panel_b_summary_path = args.panel_b_reduction / "summary.json"
    panel_b_curves_path = args.panel_b_reduction / "binned_curves.csv"
    tuning_release_path = args.tuning_release_audit / "release_audit.json"
    tuning_contract_path = args.tuning_contract / "summary.json"
    rucci_path = args.rucci_ensemble / "summary.json"
    passband_path = args.passband_comparison / "summary.json"
    trajectory_path = args.stage_trajectory / "summary.json"
    figure_summary_path = args.figure_dir / "summary.json"
    figure_path = args.figure_dir / "figure4.pdf"

    panel_a = load_json(panel_a_path)
    panel_b = load_json(panel_b_summary_path)
    panel_b_curves = pd.read_csv(panel_b_curves_path)
    tuning = load_json(tuning_release_path)
    contract = load_json(tuning_contract_path)
    rucci = load_json(rucci_path)
    passband = load_json(passband_path)
    trajectory = load_json(trajectory_path)
    figure = load_json(figure_summary_path)

    checkpoint = str(production["checkpoint_sha256"])
    claimed_checkpoints = (
        panel_a.get("checkpoint_sha256"),
        panel_b.get("checkpoint_sha256"),
        tuning.get("source_provenance", {}).get("checkpoint_sha256"),
        contract.get("checkpoint_sha256"),
        passband.get("checkpoint_sha256"),
        trajectory.get("checkpoint_sha256"),
        figure.get("checkpoint_sha256"),
    )
    if any(str(value) != checkpoint for value in claimed_checkpoints):
        raise ValueError("manuscript source artifacts do not share one checkpoint")

    selected = panel_a["selected"]
    lowest = panel_b_curves.loc[panel_b_curves.bin_index.eq(0)].set_index("outcome")
    reference = panel_b_curves.loc[panel_b_curves.bin_index.eq(5)].set_index("outcome")
    contract_files = contract.get("files", {})
    fit_record = contract_files.get(
        "all_yu_fits",
        contract_files.get(
            "all_fits", args.tuning_contract / "all_validated_yu_fits.csv"
        ),
    )
    fit_path = Path(fit_record)
    if not fit_path.is_absolute():
        fit_path = args.tuning_contract / fit_path.name
    if not fit_path.is_file():
        raise FileNotFoundError(fit_path)
    fit_table = pd.read_csv(fit_path)
    report = {
        "analysis": "Figure 4 results-text chain of custody",
        "release_ready": True,
        "checkpoint_sha256": checkpoint,
        "figure_pdf": source(figure_path),
        "production_audit": source(args.production_audit),
        "sources": {
            "panel_a": source(panel_a_path),
            "panel_b_summary": source(panel_b_summary_path),
            "panel_b_curves": source(panel_b_curves_path),
            "tuning_release": source(tuning_release_path),
            "tuning_contract": source(tuning_contract_path),
            "rucci_ensemble": source(rucci_path),
            "passband_comparison": source(passband_path),
            "stage_trajectory": source(trajectory_path),
            "figure_summary": source(figure_summary_path),
        },
        "claims": {
            "panel_a": {
                "motion_rate_spikes_s": selected["motion_rate_spikes_s"],
                "stabilized_rate_spikes_s": selected["stable_rate_spikes_s"],
                "motion_ssi_bits_per_spike": selected["motion_ssi_bits_per_spike"],
                "stabilized_ssi_bits_per_spike": selected["stable_ssi_bits_per_spike"],
                "illustrative_not_inferential": panel_a["response_selection"][
                    "illustrative_not_inferential"
                ],
            },
            "panel_b": {
                "n_units": panel_b["n_units"],
                "n_images": panel_b["n_images"],
                "n_traces": panel_b["n_traces"],
                "lowest_path_bin": lowest[
                    ["x_median", "effect_percent", "ci_low", "ci_high"]
                ].to_dict(orient="index"),
                "reference_path_bin": reference[
                    ["x_median", "effect_percent", "ci_low", "ci_high"]
                ].to_dict(orient="index"),
            },
            "tuning": {
                "population_policy": figure.get("population_policy", "validated"),
                "n_acquired": tuning["n_units"],
                "n_validated": tuning["n_validated_for_figure4"],
                "n_displayed": int(figure["panels"]["E"]["n_units"]),
                "preferred_sf_range_cpd": [
                    float(fit_table.preferred_sf_cpd.min()),
                    float(fit_table.preferred_sf_cpd.max()),
                ],
                "preferred_tf_range_hz": [
                    float(fit_table.preferred_tf_hz.min()),
                    float(fit_table.preferred_tf_hz.max()),
                ],
            },
            "rucci_power": {
                "trace_filter": rucci["trace_filter"],
                "regimes": rucci["spectral_shape_regimes"],
                "equal_dynamic_mass": rucci["spectral_shape_regimes"][
                    "map_integrals_after_normalization"
                ],
                "maximum_conservation_error": rucci["complete_power_budget"][
                    "maximum_static_plus_dynamic_error"
                ],
            },
            "passband_vs_path_length": passband["outcomes"],
            "gain_invariant_stage_trajectory": {
                "n_images": trajectory["n_images"],
                "n_traces": trajectory["n_traces"],
                "n_movies": trajectory["n_image_trace_pairs"],
                "normalization": trajectory["normalization"],
                "final_stage_is_ordinary_model": trajectory[
                    "readout_trajectory"
                ]["final_stage_is_ordinary_model"],
                "temporal_modulation": trajectory["plot"][
                    "temporal modulation"
                ],
                "spatial_sharpening": trajectory["plot"][
                    "spatial sharpening"
                ],
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
