#!/usr/bin/env python3
"""Fail-closed production audit for the revised Figure 4.

This audit does not recompute scientific effects.  It verifies that the
rendered A--H figure consumed the released artifacts that define those effects,
that each panel stays inside its stated claim boundary, and that the tuning
examples preserve exact source-unit identity from raw grid through fitted
passband.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--panel-a-audit", type=Path, required=True)
    parser.add_argument("--panel-b-reduction", type=Path, required=True)
    parser.add_argument("--tuning-release-audit", type=Path, required=True)
    parser.add_argument("--tuning-visual-audit", type=Path, required=True)
    parser.add_argument("--tuning-contract", type=Path, required=True)
    parser.add_argument(
        "--example-contract",
        type=Path,
        default=None,
        help="Validated exemplar contract; defaults to --tuning-contract.",
    )
    parser.add_argument(
        "--population-spec",
        type=Path,
        default=None,
        help="Explicit exact-unit population NPZ for an all-unit release.",
    )
    parser.add_argument("--rucci-ensemble", type=Path, required=True)
    parser.add_argument("--passband-comparison", type=Path, required=True)
    parser.add_argument(
        "--population-shards", type=Path, nargs="+", required=True
    )
    parser.add_argument(
        "--spectral-replay-tuning-table",
        type=Path,
        required=True,
        help="Exact tuning table used when the cached spectral replay was built.",
    )
    parser.add_argument("--stage-trajectory", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_visual_audit_pages(tuning_release: dict) -> list[int]:
    """Resolve the complete atlas-page contract for this model's valid units."""
    declared = tuning_release.get("visual_audit_contract", {}).get(
        "expected_validated_atlas_pages"
    )
    if declared is not None:
        pages = [int(value) for value in declared]
    else:
        n_units = int(tuning_release.get("n_validated_for_figure4", 0))
        units_per_page = 20
        pages = list(range(1, (n_units + units_per_page - 1) // units_per_page + 1))
    if not pages or pages != list(range(1, len(pages) + 1)):
        raise ValueError("tuning release contains an invalid visual-audit page contract")
    return pages


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_summary = load_json(args.figure_dir / "summary.json")
    panel_a = load_json(args.panel_a_audit / "summary.json")
    panel_b = load_json(args.panel_b_reduction / "summary.json")
    tuning_release = load_json(args.tuning_release_audit / "release_audit.json")
    tuning_visual = load_json(args.tuning_visual_audit)
    tuning_contract = load_json(args.tuning_contract / "summary.json")
    example_dir = args.example_contract or args.tuning_contract
    example_contract = load_json(example_dir / "crossed_example_summary.json")
    rucci = load_json(args.rucci_ensemble / "summary.json")
    passband = load_json(args.passband_comparison / "summary.json")
    trajectory = load_json(args.stage_trajectory / "summary.json")
    shard_summaries = [
        load_json(path.parent / "summary.json") for path in args.population_shards
    ]
    examples = pd.read_csv(example_dir / "crossed_example_fits.csv")
    unit_audit = pd.read_csv(args.tuning_release_audit / "unit_measurement_audit.csv")
    tuning_table = pd.read_csv(args.tuning_contract / "frequency_tuning_grouped.csv")
    replay_tuning_table = pd.read_csv(args.spectral_replay_tuning_table)
    contract_files = tuning_contract.get("files", {})
    all_fits_record = contract_files.get(
        "all_yu_fits",
        contract_files.get(
            "all_fits", args.tuning_contract / "all_validated_yu_fits.csv"
        ),
    )
    all_fits_path = Path(all_fits_record)
    if not all_fits_path.is_absolute():
        all_fits_path = args.tuning_contract / all_fits_path.name
    if not all_fits_path.is_file():
        raise FileNotFoundError(all_fits_path)
    population_fits = pd.read_csv(all_fits_path)

    checkpoint = str(figure_summary["checkpoint_sha256"])
    population_policy = str(figure_summary.get("population_policy", "validated"))
    if population_policy not in ("validated", "all_checkpoint_available"):
        raise ValueError(f"unsupported Figure-4 population policy: {population_policy}")
    n_available_units = int(tuning_release.get("n_units", 0))
    n_validated_units = int(tuning_release.get("n_validated_for_figure4", 0))
    expected_atlas_pages = expected_visual_audit_pages(tuning_release)
    n_analysis_units = (
        n_available_units
        if population_policy == "all_checkpoint_available"
        else n_validated_units
    )
    source_digests = {
        "Panel A": panel_a.get("checkpoint_sha256"),
        "Panel B": panel_b.get("checkpoint_sha256"),
        "tuning release": tuning_release.get("source_provenance", {}).get(
            "checkpoint_sha256"
        ),
        "tuning contract": tuning_contract.get("checkpoint_sha256"),
        "passband comparison": passband.get("checkpoint_sha256"),
        "Panel H": trajectory.get("checkpoint_sha256"),
    }
    gates: list[dict[str, object]] = []

    def gate(panel: str, name: str, passed: bool, evidence: object) -> None:
        gates.append(
            {
                "panel": panel,
                "gate": name,
                "passed": bool(passed),
                "evidence": evidence,
            }
        )

    gate(
        "global",
        "one checkpoint across all model-derived panels",
        all(str(value) == checkpoint for value in source_digests.values()),
        source_digests,
    )
    gate(
        "global",
        "locked 12 by 10 inch A--H layout",
        figure_summary.get("page_size_inches") == [12.0, 10.0]
        and set(figure_summary.get("panel_layout_inches", {})) == set("ABCDEFGH"),
        figure_summary.get("page_size_inches"),
    )
    canonical_outputs = {
        suffix: args.figure_dir / f"figure4.{suffix}"
        for suffix in ("pdf", "png", "svg")
    }
    gate(
        "global",
        "canonical PDF, PNG, and SVG release files exist",
        all(path.exists() and path.stat().st_size > 0 for path in canonical_outputs.values()),
        {
            suffix: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
            for suffix, path in canonical_outputs.items()
        },
    )
    gate(
        "A",
        "real filtered fixation and lag-aligned stabilized counterfactual",
        "low-pass-filtered" in str(panel_a.get("trace_filtering", ""))
        and panel_a.get("trace_window_selection", {}).get("alignment_kind")
        == "resolved_model_mean_peak_lag",
        {
            "trace_filtering": panel_a.get("trace_filtering"),
            "alignment": panel_a.get("trace_window_selection", {}).get("alignment"),
        },
    )
    selected_a = panel_a.get("selected", {})
    gate(
        "A",
        "illustrative unit increases both rate and SSI within disclosed bounds",
        float(selected_a.get("rate_change_percent", -np.inf)) > 0
        and float(selected_a.get("ssi_change_percent", -np.inf)) > 0
        and bool(panel_a.get("response_selection", {}).get("illustrative_not_inferential")),
        {
            "rate_change_percent": selected_a.get("rate_change_percent"),
            "ssi_change_percent": selected_a.get("ssi_change_percent"),
            "n_scored_combinations": panel_a.get("n_scored_combinations"),
        },
    )
    gate(
        "B",
        "all exact checkpoint-available units and all filtered fixations pooled",
        int(panel_b.get("n_units", 0)) == n_available_units
        and int(panel_b.get("n_images", 0)) == 40
        and int(panel_b.get("n_traces", 0)) == 200
        and "no microsaccade stratification" in str(panel_b.get("fixation_selection", "")),
        {
            "n_units": panel_b.get("n_units"),
            "n_images": panel_b.get("n_images"),
            "n_traces": panel_b.get("n_traces"),
            "selection": panel_b.get("fixation_selection"),
        },
    )
    panel_b_render = figure_summary["panels"]["B"]
    gate(
        "B",
        "unit heterogeneity is shown in addition to pooled uncertainty",
        "per-unit boxes" in str(panel_b_render.get("distribution_display", ""))
        and "5th--95th percentiles" in str(
            panel_b_render.get("distribution_display", "")
        )
        and isinstance(panel_b.get("unit_distribution"), dict)
        and all(
            int(
                panel_b_render.get("curves", {})
                .get(outcome, {})
                .get("unit_distribution", {})
                .get("n_units", 0)
            )
            == n_available_units
            for outcome in ("rate", "SSI")
        ),
        {
            "display": panel_b_render.get("distribution_display"),
            "source_contract": panel_b.get("unit_distribution"),
        },
    )
    gate(
        "C",
        "conditional spectra compare shape at equal dynamic-power mass",
        figure_summary["panels"]["C"].get("data_dependent") is True
        and figure_summary["panels"]["C"].get("passband_contours_drawn") is False
        and bool(figure_summary["panels"]["C"].get(
            "equal_dynamic_mass_before_comparison"
        ))
        and figure_summary["panels"]["C"].get(
            "selected_by_microsaccade_label"
        )
        is False
        and len(figure_summary["panels"]["C"].get("regimes", [])) == 2
        and bool(rucci.get("spectral_shape_regimes", {}).get(
            "selection_reads_neural_or_model_responses"
        ))
        is False
        and np.allclose(
            rucci.get("spectral_shape_regimes", {}).get(
                "map_integrals_after_normalization", []
            ),
            1.0,
            atol=1e-6,
            rtol=0.0,
        ),
        figure_summary["panels"]["C"],
    )

    validated = unit_audit.loc[unit_audit.validated_for_figure4.astype(bool)]
    source_lookup = unit_audit.set_index("unit_index")
    example_evidence = []
    examples_ok = True
    for row in examples.itertuples(index=False):
        source = int(row.source_unit_index)
        audit_row = source_lookup.loc[source]
        if population_policy == "all_checkpoint_available":
            population_rows = population_fits.loc[
                population_fits.source_unit_index.eq(source)
            ]
            if len(population_rows) != 1:
                raise ValueError(
                    "all-unit tuning contract does not contain exactly one row for "
                    f"example source unit {source}"
                )
            population_row = population_rows.iloc[0]
            display_unit = int(population_row.unit_index)
            parameter_columns = (
                "preferred_sf_cpd",
                "preferred_tf_hz",
                "sigma_s",
                "zeta_s",
                "sigma_t",
                "zeta_t",
                "q",
            )
            parameter_identity = bool(
                np.allclose(
                    [float(getattr(row, column)) for column in parameter_columns],
                    [float(population_row[column]) for column in parameter_columns],
                    atol=1e-12,
                    rtol=0.0,
                )
            )
        else:
            display_unit = int(row.unit_index)
            parameter_identity = True
        table_sources = tuning_table.loc[
            tuning_table.unit_index.eq(display_unit), "source_unit_index"
        ].drop_duplicates()
        identity_ok = len(table_sources) == 1 and int(table_sources.iloc[0]) == source
        role_group_ok = (
            str(row.role) == "low SF / high TF"
            and str(audit_row.crossed_group) == "low recorded SF / high twin TF"
        ) or (
            str(row.role) == "high SF / low TF"
            and str(audit_row.crossed_group) == "high recorded SF / low twin TF"
        )
        row_ok = (
            identity_ok
            and parameter_identity
            and bool(audit_row.validated_for_figure4)
            and str(row.response_column)
            == "preferred_direction_delta_f0_expected_count"
            and float(row.full_support_r2) >= 0.8
            and role_group_ok
        )
        examples_ok &= row_ok
        example_evidence.append(
            {
                "display_unit_index": display_unit,
                "source_unit_index": source,
                "canonical_channel": int(row.canonical_channel),
                "role": str(row.role),
                "audited_group": str(audit_row.crossed_group),
                "fit_r2": float(row.full_support_r2),
                "identity_exact": identity_ok,
                "fit_parameters_exact": parameter_identity,
            }
        )
    gate(
        "D",
        "raw surfaces, fitted lassos, and labels share exact source identity",
        examples_ok,
        example_evidence,
    )
    gate(
        "D",
        "example selection is deterministic and response-independent",
        bool(example_contract.get("release_ready"))
        and example_contract.get("selection_reads_retinal_motion_response") is False
        and sorted(example_contract.get("unit_indices", []))
        == sorted(examples.unit_index.astype(int).tolist()),
        {
            "selection": example_contract.get("selection"),
            "reads_retinal_motion_response": example_contract.get(
                "selection_reads_retinal_motion_response"
            ),
            "unit_indices": example_contract.get("unit_indices"),
        },
    )
    gate(
        "C/F",
        "eye motion redistributes rather than creates image power",
        bool(rucci.get("complete_power_budget", {}).get(
            "all_power_is_redistributed_not_created"
        ))
        and float(
            rucci.get("complete_power_budget", {}).get(
                "maximum_static_plus_dynamic_error", np.inf
            )
        )
        < 1e-12,
        rucci.get("complete_power_budget"),
    )
    gate(
        "E",
        "declared SFxTF population is complete and identity preserving",
        n_analysis_units > 0
        and int(figure_summary["panels"]["E"].get("n_units", -1))
        == n_analysis_units
        and int(figure_summary["panels"]["E"].get("n_passband_contours", -1))
        == n_analysis_units
        and figure_summary["panels"]["E"].get(
            "contours_share_authoritative_code_with_panels_d_and_f"
        )
        is True
        and (
            (
                population_policy == "validated"
                and len(validated) == n_validated_units
            )
            or (
                population_policy == "all_checkpoint_available"
                and tuning_contract.get("population_policy")
                == "all_checkpoint_available"
                and bool(
                    tuning_contract.get("gates", {}).get(
                        "all_checkpoint_available_units_included_once"
                    )
                )
                and bool(
                    tuning_contract.get("gates", {}).get(
                        "every_exported_passband_surface_finite"
                    )
                )
            )
        ),
        {
            "population_policy": population_policy,
            "n_acquired": tuning_release.get("n_units"),
            "n_validated": tuning_release.get("n_validated_for_figure4"),
            "n_contours": figure_summary["panels"]["E"].get(
                "n_passband_contours"
            ),
            "failure_counts": tuning_release.get("failure_counts"),
        },
    )
    gate(
        "D/E",
        "displayed raw surfaces passed visual audit and population fits are finite",
        str(tuning_visual.get("status")) == "PASS"
        and int(tuning_visual.get("n_units_inspected", 0)) == n_validated_units
        and tuning_visual.get("atlas_pages_inspected") == expected_atlas_pages,
        {
            "status": tuning_visual.get("status"),
            "n_units_inspected": tuning_visual.get("n_units_inspected"),
            "pages": tuning_visual.get("atlas_pages_inspected"),
            "expected_pages": expected_atlas_pages,
        },
    )
    panel_g = figure_summary["panels"]["G"]
    gate(
        "G",
        "direct passband association uses the declared SFxTF population",
        int(panel_g.get("n_units", -1)) == n_analysis_units
        and int(passband.get("n_units", -1)) == n_analysis_units
        and str(passband.get("population_policy", "validated"))
        == population_policy
        and "direct measured-motion minus stabilized" in str(
            panel_g.get("y_definition", "")
        )
        and panel_g.get("fraction_of_total_effect_explained") is False
        and "not variance explained" in str(panel_g.get("claim_boundary", "")),
        {
            "figure_n_units": panel_g.get("n_units"),
            "comparison_n_units": passband.get("n_units"),
            "rate_spearman": panel_g.get("rate_percent", {}).get("spearman_rho"),
            "ssi_spearman": panel_g.get("ssi_percent", {}).get("spearman_rho"),
            "claim_boundary": panel_g.get("claim_boundary"),
        },
    )
    spectral_columns = (
        "unit_index",
        "spatial_cpd",
        "temporal_hz",
        "probe_orientation_deg",
        "response_amp_rms",
        "passband_weight",
        "source_unit_index",
        "canonical_channel",
        "session",
        "cid",
    )
    spectral_contract_ok = len(tuning_table) == len(replay_tuning_table)
    spectral_differences: dict[str, object] = {}
    for column in spectral_columns:
        if column not in tuning_table or column not in replay_tuning_table:
            spectral_contract_ok = False
            spectral_differences[column] = "missing"
            continue
        released = tuning_table[column]
        replayed = replay_tuning_table[column]
        if pd.api.types.is_numeric_dtype(released):
            maximum_error = float(
                np.nanmax(
                    np.abs(
                        released.to_numpy(dtype=float)
                        - replayed.to_numpy(dtype=float)
                    )
                )
            )
            column_ok = maximum_error <= 1e-12
            spectral_differences[column] = maximum_error
        else:
            column_ok = bool(
                np.array_equal(
                    released.astype(str).to_numpy(),
                    replayed.astype(str).to_numpy(),
                )
            )
            spectral_differences[column] = column_ok
        spectral_contract_ok &= column_ok
    gate(
        "G",
        "cached spectral replay used the released SFxTF passband tensors",
        spectral_contract_ok,
        spectral_differences,
    )
    if args.population_spec is not None:
        released_population_path = args.population_spec
    else:
        released_population_path = Path(
            tuning_contract["files"]["population_spec_npz"]
        )
    if not released_population_path.is_file():
        raise FileNotFoundError(released_population_path)
    released_population_sha = sha256(released_population_path)
    replay_population_shas = [
        str(
            summary.get("model_provenance", {}).get(
                "population_spec_npz_sha256", ""
            )
        )
        for summary in shard_summaries
    ]
    gate(
        "G",
        "cached spectral replay used the released exact-unit population",
        len(shard_summaries) >= 1
        and all(value == released_population_sha for value in replay_population_shas)
        and all(
            int(summary.get("n_units", 0)) == n_analysis_units
            for summary in shard_summaries
        )
        and all(int(summary.get("n_traces", 0)) == 200 for summary in shard_summaries)
        and sum(int(summary.get("n_images", 0)) for summary in shard_summaries) == 40,
        {
            "released_population_sha256": released_population_sha,
            "replay_population_sha256": replay_population_shas,
            "n_images_total": sum(
                int(summary.get("n_images", 0)) for summary in shard_summaries
            ),
            "n_traces": [summary.get("n_traces") for summary in shard_summaries],
            "n_units": [summary.get("n_units") for summary in shard_summaries],
        },
    )
    gate(
        "H",
        "cumulative readout reconstructs the ordinary model and cached G output",
        bool(
            trajectory.get("readout_trajectory", {}).get(
                "final_stage_is_ordinary_model", False
            )
        )
        and not bool(
            trajectory.get("readout_trajectory", {}).get(
                "synthetic_reference", False
            )
        )
        and not bool(
            trajectory.get("readout_trajectory", {}).get(
                "affine_or_tangent_model", False
            )
        )
        and float(
            trajectory.get("identity_checks", {}).get(
                "ordinary_output_max_abs", np.inf
            )
        )
        < 2e-5
        and float(
            trajectory.get("cached_G_output_checks", {}).get(
                "ssi_bits_per_spike_max_abs", np.inf
            )
        )
        < 2e-4,
        {
            "n_movies": trajectory.get("n_image_trace_pairs"),
            "identity_checks": trajectory.get("identity_checks"),
            "cached_G_output_checks": trajectory.get("cached_G_output_checks"),
        },
    )
    panel_h = figure_summary["panels"]["H"]
    gate(
        "H",
        "stage effects are gain invariant and expressed in natural units",
        "movie-wide mean" in str(panel_h.get("normalization", ""))
        and "mean-normalized spatial rate map" in str(
            panel_h.get("normalization", "")
        )
        and panel_h.get("mean_rate_gain_plotted") is False
        and panel_h.get("final_stage_is_ordinary_model") is True
        and len(panel_h.get("cumulative_stage_labels", [])) == 3
        and len(panel_h.get("temporal_modulation", {}).get("center", [])) == 3
        and len(panel_h.get("spatial_sharpening", {}).get("center", [])) == 3,
        {
            "normalization": panel_h.get("normalization"),
            "visualization": panel_h.get("visualization"),
            "mean_rate_gain_plotted": panel_h.get("mean_rate_gain_plotted"),
        },
    )
    gate(
        "H",
        "stage-trajectory inference uses at least 100 crossed movies",
        int(trajectory.get("n_image_trace_pairs", 0)) >= 100
        and bool(figure_summary.get("panel_h", {}).get("inference_ready")),
        {
            "n_images": trajectory.get("n_images"),
            "n_traces": trajectory.get("n_traces"),
            "n_movies": trajectory.get("n_image_trace_pairs"),
        },
    )
    gate(
        "H",
        "population-output stage uses the released tuning population",
        int(trajectory.get("population_n_units", 0)) == n_analysis_units
        and int(figure_summary.get("panel_populations", {}).get("H", {}).get(
            "n_units", 0
        ))
        == n_analysis_units,
        {
            "trajectory_population_n_units": trajectory.get("population_n_units"),
            "figure_population_n_units": figure_summary.get(
                "panel_populations", {}
            ).get("H", {}).get("n_units"),
        },
    )

    all_pass = all(bool(item["passed"]) for item in gates)
    report = {
        "analysis": "revised Figure-4 fail-closed production audit",
        "release_ready": all_pass,
        "checkpoint_sha256": checkpoint,
        "population_policy": population_policy,
        "n_analysis_units": n_analysis_units,
        "figure_png": str((args.figure_dir / "figure4.png").resolve()),
        "figure_png_sha256": sha256(args.figure_dir / "figure4.png"),
        "gates": gates,
    }
    (args.out_dir / "production_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Revised Figure 4 production audit",
        "",
        f"Overall: **{'PASS' if all_pass else 'FAIL'}**",
        "",
        "| Panel | Gate | Status |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {item['panel']} | {item['gate']} | {'PASS' if item['passed'] else 'FAIL'} |"
        for item in gates
    )
    (args.out_dir / "production_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(args.out_dir / "production_audit.json")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
