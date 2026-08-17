from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.fig4.mechanism_audit_v1.registration_mechanism import plot_semantics as plot


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def marker(path: Path, stage: str, products: list[Path], basis_path: Path) -> None:
    fold = int(path.parts[-4].split("_", 1)[1])
    contrast = path.parts[-3]
    write_json(
        path,
        {
            "schema_version": "fig4-registration-pq-semantics-v1",
            "complete": True,
            "stage": stage,
            "fold": fold,
            "contrast": contrast,
            "basis_path": str(basis_path),
            "basis_sha256": sha(basis_path),
            "input_fingerprint": {"synthetic_fixture_revision": 1},
            "products": [
                {"path": str(product), "sha256": sha(product), "size_bytes": product.stat().st_size}
                for product in products
            ],
        },
    )


def make_finalized_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    leverage_rows = []
    variance_rows = []
    readout_rows = []
    unit_rows = []
    supporting = []
    projectors = []
    low_units = list(range(71))
    high_units = list(range(71, 100))
    for fold in plot.FOLDS:
        for contrast_i, contrast in enumerate(plot.CONTRAST_ORDER):
            rng = np.random.default_rng(10_000 + fold * 100 + contrast_i)
            basis, _ = np.linalg.qr(rng.normal(size=(128, 8)), mode="reduced")
            product_dir = root / "rank8_validation/fold_products" / f"fold_{fold}" / contrast
            product_dir.mkdir(parents=True, exist_ok=True)
            basis_path = product_dir / "U.npy"
            projector_path = product_dir / "P.npy"
            predictions_path = product_dir / "test_predictions.npz"
            np.save(basis_path, basis.astype(np.float32))
            np.save(projector_path, (basis @ basis.T).astype(np.float32))
            np.savez_compressed(predictions_path, heldout=np.asarray([0.8], dtype=np.float32))
            projectors.append(
                {
                    "fold": fold,
                    "contrast": contrast,
                    "rank": 8,
                    "basis_path": str(basis_path),
                    "basis_sha256": sha(basis_path),
                    "projector_path": str(projector_path),
                    "projector_sha256": sha(projector_path),
                    "heldout_predictions_path": str(predictions_path),
                    "heldout_predictions_sha256": sha(predictions_path),
                }
            )
            leverage = np.square(basis).sum(axis=1)
            order = np.argsort(-leverage)
            neff = leverage.sum() ** 2 / np.square(leverage).sum()
            for rank, channel in enumerate(order, start=1):
                leverage_rows.append(
                    {
                        "fold": fold,
                        "contrast": contrast,
                        "native_channel": channel,
                        "leverage_rank_descending": rank,
                        "leverage_score_p_cc": leverage[channel],
                        "cumulative_leverage_fraction_at_channel_rank": leverage[order[:rank]].sum() / 8,
                        "effective_participating_channel_count": neff,
                    }
                )
            target_scale = plot.TARGET_SCALE[contrast]
            for analysis, scale, p_fraction in (
                ("stabilized_visual_content_image_means", 0.0, 0.24 + 0.01 * fold),
                ("movement_change_from_stabilization", target_scale, 0.52 + 0.02 * fold),
            ):
                q_fraction = 1 - p_fraction
                q_per_dim = 0.2
                ratio = 15.0 * p_fraction / q_fraction
                variance_rows.append(
                    {
                        "fold": fold,
                        "contrast": contrast,
                        "analysis": analysis,
                        "scale": scale,
                        "candidate_p_total_fraction": p_fraction,
                        "complementary_q_total_fraction": q_fraction,
                        "candidate_p_energy_per_dimension": ratio * q_per_dim,
                        "complementary_q_energy_per_dimension": q_per_dim,
                        "p_to_q_per_dimension_energy_ratio": ratio,
                    }
                )
            population = plot.TARGET_POPULATION[contrast]
            n_units = plot.TARGET_UNITS[contrast]
            for component, value in (("candidate_p_content", 0.42), ("complementary_q_content", 0.78)):
                readout_rows.append(
                    {
                        "fold": fold,
                        "contrast": contrast,
                        "analysis": "baseline_visual_reconstruction",
                        "scale": 0.0,
                        "scale_a": np.nan,
                        "scale_b": np.nan,
                        "population": population,
                        "n_units": n_units,
                        "component": component,
                        "primary_map_recovery_weighting": "paired expected spikes",
                        "normalized_map_recovery_vs_training_mean_r2": value + 0.01 * fold,
                        "normalized_map_movement_effect_recovery_r2": np.nan,
                    }
                )
            for component, value in (
                ("candidate_p_only_contrast", 0.81 - 0.03 * contrast_i),
                ("complementary_q_only_contrast", -0.12 + 0.04 * contrast_i),
            ):
                readout_rows.append(
                    {
                        "fold": fold,
                        "contrast": contrast,
                        "analysis": "defining_contrast_movement_effect_decomposition",
                        "scale": plot.TARGET_SCALE[contrast],
                        "population": population,
                        "n_units": n_units,
                        "component": component,
                        "scale_a": {"low_0_to_2": 0.0, "high_0_to_1": 0.0, "high_1_to_3": 1.0}[
                            contrast
                        ],
                        "scale_b": plot.TARGET_SCALE[contrast],
                        "primary_map_recovery_weighting": "paired expected spikes",
                        "normalized_map_recovery_vs_training_mean_r2": np.nan,
                        "normalized_map_movement_effect_recovery_r2": value + 0.01 * fold,
                    }
                )
            units = low_units if contrast == "low_0_to_2" else high_units
            for unit in units:
                activity = 0.12 + 0.75 * (unit / 99) + 0.01 * fold
                benefit = 0.01 * (unit - np.mean(units)) + 0.002 * fold
                reversal = -0.008 * (unit - np.mean(units)) + 0.001 * fold
                unit_rows.append(
                    {
                        "fold": fold,
                        "contrast": contrast,
                        "unit_index": unit,
                        "historical_sf_population": population,
                        "sf_split_metric": 0.1 + 0.04 * unit,
                        "weight_based_candidate_p_reliance": min(0.98, activity + 0.03),
                        "activity_activity_weighted_p_reliance_excluding_covariance": min(0.99, activity),
                        "observed_ssi_benefit_for_projector_contrast_bits": benefit,
                        "observed_ssi_1_to_3_change_bits": reversal,
                        "high_motion_reversal": reversal < 0,
                        "observational_ssi_scope": (
                            f"fold_{fold}_crossed_test_block_only_2_images_x_6_trajectories_x_40_frames"
                        ),
                    }
                )

            stage_root = root / "pq_intermediate" / f"fold_{fold}" / contrast
            variance_dir = stage_root / "variance"
            readout_dir = stage_root / "readout"
            probes_dir = stage_root / "probes"
            variance_dir.mkdir(parents=True, exist_ok=True)
            readout_dir.mkdir(parents=True, exist_ok=True)
            probes_dir.mkdir(parents=True, exist_ok=True)
            variance_paths = [
                variance_dir / "native_channel_leverage.csv",
                variance_dir / "pq_variance_decomposition.csv",
                variance_dir / "variance_supporting_arrays.npz",
            ]
            pd.DataFrame([{"ok": 1}]).to_csv(variance_paths[0], index=False)
            pd.DataFrame([{"ok": 1}]).to_csv(variance_paths[1], index=False)
            np.savez_compressed(variance_paths[2], basis=basis.astype(np.float32))
            marker(variance_dir / "complete.json", "variance", variance_paths, basis_path)
            readout_paths = [
                readout_dir / "pq_readout_decomposition.csv",
                readout_dir / "per_unit_pq_reliance.csv",
                readout_dir / "pq_readout_supporting_arrays.npz",
                readout_dir / "readout_numerical_diagnostics.json",
            ]
            pd.DataFrame([{"ok": 1}]).to_csv(readout_paths[0], index=False)
            pd.DataFrame([{"ok": 1}]).to_csv(readout_paths[1], index=False)
            np.savez_compressed(readout_paths[2], recovery=np.asarray([0.8]))
            write_json(readout_paths[3], {"passed": True})
            marker(readout_dir / "complete.json", "readout", readout_paths, basis_path)
            probe_paths = [
                probes_dir / "pq_descriptive_probes.csv",
                probes_dir / "probe_feature_cache_reference.json",
            ]
            pd.DataFrame([{"ok": 1}]).to_csv(probe_paths[0], index=False)
            write_json(probe_paths[1], {"ok": True})
            marker(probes_dir / "complete.json", "probes", probe_paths, basis_path)

            supporting_dir = root / "pq_supporting_arrays"
            supporting_dir.mkdir(parents=True, exist_ok=True)
            for stage, source in (("variance", variance_paths[2]), ("readout", readout_paths[2])):
                path = supporting_dir / f"fold_{fold}__{contrast}__{stage}.npz"
                path.write_bytes(source.read_bytes())
                supporting.append(
                    {
                        "path": str(path),
                        "source": str(source),
                        "sha256": sha(path),
                        "size_bytes": path.stat().st_size,
                    }
                )

    canonical_frames = {
        "native_channel_leverage.csv": pd.DataFrame(leverage_rows),
        "pq_variance_decomposition.csv": pd.DataFrame(variance_rows),
        "pq_readout_decomposition.csv": pd.DataFrame(readout_rows),
        "per_unit_pq_reliance.csv": pd.DataFrame(unit_rows),
    }
    canonical_records = {}
    for name, frame in canonical_frames.items():
        path = root / name
        frame.to_csv(path, index=False)
        canonical_records[name] = {
            "path": str(path),
            "rows": len(frame),
            "exists": True,
            "sha256": sha(path),
        }
    write_json(
        root / "pq_supporting_arrays" / "manifest.json",
        {
            "schema_version": "fig4-registration-pq-semantics-v1",
            "products": supporting,
        },
    )
    write_json(
        root / "pq_semantics_manifest.json",
        {
            "schema_version": "fig4-registration-pq-semantics-v1",
            "status": "complete",
            "complete_all_four_folds": True,
            "no_core_replay": True,
            "model_imported": False,
            "missing_basis_fold_contrasts": [],
            "missing_variance_fold_contrasts": [],
            "missing_readout_fold_contrasts": [],
            "missing_probe_fold_contrasts": [],
            "n_supporting_array_products": 24,
            "canonical_products": canonical_records,
        },
    )

    rank_dir = root / "rank8_validation"
    fold_results = []
    for fold, contrast in sorted(plot.EXPECTED_CELLS):
        for method in ("learned", "readout_svd"):
            fold_results.append(
                {
                    "fold": fold,
                    "contrast": contrast,
                    "method": method,
                    "complete_map_recovery_mean": 0.8 if method == "learned" else 0.7,
                }
            )
    pd.DataFrame(fold_results).to_csv(root / "fold_rank8_results.csv", index=False)
    write_json(
        root / "rank8_validation_gate.json",
        {
            "status": "complete",
            "rank_fixed_prospectively": 8,
            "stop_downstream_mechanism_audit": False,
            "heldout_generalization": {
                contrast: {
                    "complete": True,
                    "generalizes": True,
                    "learned_outperforms_readout_consistently": False,
                }
                for contrast in plot.CONTRAST_ORDER
            },
        },
    )
    write_json(
        rank_dir / "analysis_manifest.json",
        {
            "schema_version": "fig4-registration-rank8-validation-v1",
            "rank": 8,
            "folds": list(plot.FOLDS),
            "contrasts": list(plot.CONTRAST_ORDER),
            "uses_saved_cache_only_for_evaluation": True,
            "consensus_is_visualization_only": True,
        },
    )
    write_json(
        rank_dir / "rank8_projector_inventory.json",
        {
            "schema_version": "fig4-registration-rank8-projector-inventory-v1",
            "rank": 8,
            "consensus_for_visualization_only": True,
            "fold_projectors": projectors,
        },
    )
    return root


def test_loader_fails_closed_for_incomplete_manifest(tmp_path: Path) -> None:
    root = make_finalized_fixture(tmp_path / "input")
    manifest_path = root / "pq_semantics_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["complete_all_four_folds"] = False
    write_json(manifest_path, manifest)
    with pytest.raises(plot.DataUnavailable, match="not complete"):
        plot.load_finalized_inputs(root)


def test_loader_rejects_tampered_canonical_product(tmp_path: Path) -> None:
    root = make_finalized_fixture(tmp_path / "input")
    with (root / "native_channel_leverage.csv").open("a") as handle:
        handle.write("tampered\n")
    with pytest.raises(plot.DataUnavailable, match="hash"):
        plot.load_finalized_inputs(root)


def test_loader_rejects_missing_stage_marker(tmp_path: Path) -> None:
    root = make_finalized_fixture(tmp_path / "input")
    marker_path = root / "pq_intermediate/fold_0/low_0_to_2/readout/complete.json"
    marker_path.unlink()
    with pytest.raises(plot.DataUnavailable, match="completion marker"):
        plot.load_finalized_inputs(root)


def test_loader_rejects_mixed_stage_fingerprints(tmp_path: Path) -> None:
    root = make_finalized_fixture(tmp_path / "input")
    marker_path = root / "pq_intermediate/fold_0/low_0_to_2/probes/complete.json"
    value = json.loads(marker_path.read_text())
    value["input_fingerprint"] = {"synthetic_fixture_revision": 2}
    write_json(marker_path, value)
    with pytest.raises(plot.DataUnavailable, match="fingerprint"):
        plot.load_finalized_inputs(root)


def test_loader_rejects_supporting_array_not_mirrored_from_validated_stage(tmp_path: Path) -> None:
    root = make_finalized_fixture(tmp_path / "input")
    manifest_path = root / "pq_supporting_arrays/manifest.json"
    value = json.loads(manifest_path.read_text())
    value["products"][0]["source"] = value["products"][1]["source"]
    write_json(manifest_path, value)
    with pytest.raises(plot.DataUnavailable, match="source"):
        plot.load_finalized_inputs(root)


def test_loader_rejects_rank8_basis_provenance_disagreement(tmp_path: Path) -> None:
    root = make_finalized_fixture(tmp_path / "input")
    inventory_path = root / "rank8_validation/rank8_projector_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["fold_projectors"][0]["basis_sha256"] = "0" * 64
    write_json(inventory_path, inventory)
    with pytest.raises(plot.DataUnavailable, match="finalized fold basis"):
        plot.load_finalized_inputs(root)


def test_panel_c_preserves_negative_unclipped_recovery(tmp_path: Path) -> None:
    inputs = plot.load_finalized_inputs(make_finalized_fixture(tmp_path / "input"))
    panel = plot.prepare_panel_c(inputs.readout, draws=1_000, seed=3)
    values = panel.loc[
        panel.row_type.eq("fold")
        & panel.transformation.eq("movement map")
        & panel.component.eq("complementary Q")
    ].complete_normalized_map_recovery_r2
    assert (values < 0).any()
    assert panel.recovery_values_are_unclipped.eq(True).all()


def test_panel_c_requires_literal_defining_contrast_and_weighting(tmp_path: Path) -> None:
    inputs = plot.load_finalized_inputs(make_finalized_fixture(tmp_path / "input"))
    readout = inputs.readout.copy()
    selected = readout.analysis.eq("defining_contrast_movement_effect_decomposition")
    readout.loc[selected & readout.contrast.eq("high_1_to_3"), "scale_a"] = 0.0
    with pytest.raises(plot.DataUnavailable, match="literal defining contrast"):
        plot.prepare_panel_c(readout, draws=1_000, seed=3)


def test_panel_a_rejects_arbitrary_neff_or_leverage_threshold_surrogate(tmp_path: Path) -> None:
    inputs = plot.load_finalized_inputs(make_finalized_fixture(tmp_path / "input"))
    leverage = inputs.leverage.copy()
    leverage.loc[
        leverage.fold.eq(0) & leverage.contrast.eq("low_0_to_2"),
        "effective_participating_channel_count",
    ] = 8.0
    with pytest.raises(plot.DataUnavailable, match="does not match the leverage"):
        plot.prepare_panel_a(leverage, draws=1_000, seed=2)


def test_panel_d_averages_folds_before_unit_association(tmp_path: Path) -> None:
    inputs = plot.load_finalized_inputs(make_finalized_fixture(tmp_path / "input"))
    units, associations = plot.prepare_panel_d(inputs.per_unit, draws=1_000, seed=4)
    assert units.folds_aggregated.eq(4).all()
    assert len(units.loc[units.contrast.eq("low_0_to_2")]) == 71
    assert len(units.loc[units.contrast.eq("high_0_to_1")]) == 29
    assert len(associations) == 9


def test_source_snapshot_fails_if_finalized_input_changes_after_loading(tmp_path: Path) -> None:
    inputs = plot.load_finalized_inputs(make_finalized_fixture(tmp_path / "input"))
    with (inputs.root / "pq_variance_decomposition.csv").open("a") as handle:
        handle.write("changed after loading\n")
    with pytest.raises(plot.DataUnavailable, match="changed while plotting"):
        plot._validate_source_snapshot(inputs)


def test_complete_render_exports_vectors_600dpi_png_data_and_boundary(tmp_path: Path) -> None:
    root = make_finalized_fixture(tmp_path / "input")
    output = tmp_path / "figures"
    data = tmp_path / "figure_data"
    manifest = plot.run(
        input_dir=root,
        output_dir=output,
        figure_data_dir=data,
        bootstrap_draws=1_000,
        seed=5,
    )
    for suffix in ("svg", "pdf", "png"):
        path = output / f"{plot.FIGURE_BASENAME}.{suffix}"
        assert path.is_file() and path.stat().st_size > 1_000
    from PIL import Image

    with Image.open(output / f"{plot.FIGURE_BASENAME}.png") as image:
        assert image.info["dpi"][0] == pytest.approx(600, abs=1)
    caption = (output / plot.CAPTION_NAME).read_text()
    svg = (output / f"{plot.FIGURE_BASENAME}.svg").read_text()
    assert "compact output-relevant subspaces" in caption
    assert "What the compact output-relevant subspace" in caption
    assert "What the compact output-relevant subspace" in svg
    assert "movement-specific" in caption
    assert manifest["consensus_projectors_used_for_inference"] is False
    assert manifest["state_or_readout_cache_read"] is False
    assert manifest["rank8_basis_or_projector_arrays_read"] is False
    assert {Path(record["path"]).suffix for record in manifest["source_products"]} <= {
        ".csv",
        ".json",
        ".npz",
    }
    assert len(list(data.glob("*.csv"))) == 6
    arrays_path = data / "figure1_pq_semantics_exact_arrays.npz"
    assert arrays_path.is_file()
    with np.load(arrays_path, allow_pickle=False) as arrays:
        assert arrays["panel_c_recovery"].min() < 0
        assert set(arrays["panel_b_variation"].tolist()) == {"content", "motion"}
        assert len(arrays["panel_d_unit_index"]) == 129


def test_plot_module_has_no_model_or_cache_reader_imports() -> None:
    source = Path(plot.__file__).read_text()
    for forbidden in ("import torch", "import h5py", "STATE_CACHE", "MAP_CACHE", "READOUT_CACHE"):
        assert forbidden not in source
