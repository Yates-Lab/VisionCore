from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFRESH_ALL = ROOT / "paper" / "fig4" / "refresh_all.py"
LAUNCHER = ROOT / "paper" / "fig4" / "upstream" / "run_real_trace_matrix.py"
MERGER = ROOT / "paper" / "fig4" / "upstream" / "merge_backimage_real_trace_ssi_matrix_shards.py"
STAGER = ROOT / "paper" / "fig4" / "upstream" / "stage_real_trace_source_assets.py"
SOURCE_BUNDLE = ROOT / "paper" / "fig4" / "upstream" / "build_real_trace_source_asset_bundle.py"
TRACE_BANK_METADATA_BUILDER = ROOT / "paper" / "fig4" / "upstream" / "build_trace_bank_metadata.py"
COMPARATOR = ROOT / "paper" / "fig4" / "upstream" / "compare_real_trace_matrix_outputs.py"
AUDITOR = ROOT / "paper" / "fig4" / "upstream" / "audit_real_trace_matrix_replay.py"
MATRIX_RUNNER = ROOT / "paper" / "fig4" / "upstream" / "score_real_trace_matrix.py"
BASELINE_RUNNER = ROOT / "paper" / "fig4" / "upstream" / "score_real_trace_stabilized_baseline.py"
UPSTREAM_DIR = ROOT / "paper" / "fig4" / "upstream"
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from real_trace_matrix.core import (
    score_matrix,
    speed_threshold_mad,
    trace_hash,
    trace_items_from_table_and_array,
    trace_scale_metrics,
)
import build_trace_bank_metadata as trace_bank_metadata_builder
import build_real_trace_source_asset_bundle as source_asset_bundle
import run_real_trace_matrix as real_trace_launcher
import score_real_trace_matrix as real_trace_scorer
import score_real_trace_stabilized_baseline as stabilized_baseline
from score_real_trace_matrix import filter_source_rows, parse_session_filter

RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_"
    "medoidPosthocminRepcomplete0p45_movieMedoid"
)
SOURCE_ASSET_RELS = (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv",
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/window_features.csv",
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv",
    f"outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/population_spec_{RR100_VERSION}.json",
    f"outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/population_spec_{RR100_VERSION}.npz",
    "outputs/artifacts/mcfarland/mcfarland_outputs_mono.pkl",
)


def _load_refresh_all_module():
    spec = importlib.util.spec_from_file_location("fig4_refresh_all_under_test", REFRESH_ALL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_plan(tmp_path: Path, profile: str, *extra_args: str) -> dict:
    plan_json = tmp_path / f"{profile}_plan.json"
    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--profile",
            profile,
            "--plan-json",
            str(plan_json),
            "--no-print-commands",
            *extra_args,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(plan_json.read_text(encoding="utf-8"))


def _arg_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_refresh_graph_declares_recovered_trace_bank_metadata_stage():
    refresh_all = _load_refresh_all_module()
    stage = next(stage for stage in refresh_all.STAGES if stage.key == "trace_bank_metadata")

    assert stage.script == TRACE_BANK_METADATA_BUILDER
    assert stage.out_dir_flag == "--out-dir"
    assert "--force" in stage.extra_args
    assert stage.produces == {
        "filtered_path_length_le350arcmin/trace_bank_metadata_filtered.csv":
            refresh_all._paths.TRACE_BANK_METADATA_FILTERED_CSV,
    }


def test_production_plan_records_recovered_deep_matrix_contract(tmp_path):
    manifest = _run_plan(tmp_path, "production")

    assert manifest["profile"] == "production"
    assert manifest["rr100_version"] == RR100_VERSION
    assert manifest["temporal_contract"]["model_history_frames"] == 32
    assert manifest["temporal_contract"]["scored_trace_samples"] == 40
    assert manifest["selected_shards"] == ["images_000_050", "images_050_100"]

    score_commands = [cmd for cmd in manifest["commands"] if cmd["stage"] == "score_shard"]
    assert len(score_commands) == 2
    first = score_commands[0]["argv"]
    second = score_commands[1]["argv"]

    assert first[3].endswith("paper/fig4/upstream/score_real_trace_matrix.py")
    assert _arg_after(first, "--n-images") == "100"
    assert _arg_after(first, "--n-traces") == "1000"
    assert _arg_after(first, "--seed") == "20260717"
    assert _arg_after(first, "--n-timepoints") == "40"
    assert _arg_after(first, "--bin-seconds") == "0.008333333333333333"
    assert _arg_after(first, "--patch-size-px") == "540"
    assert _arg_after(first, "--image-contrast-quantile") == "0.75"
    assert _arg_after(first, "--image-min-orientation-coherence") == "0.2"
    assert _arg_after(first, "--min-strong-contour-images") == "40"
    assert _arg_after(first, "--trace-scale-metric") == "rendered_path_length_arcmin"
    assert _arg_after(first, "--trace-sampling") == "quantile"
    assert _arg_after(first, "--min-microsaccade-traces") == "200"
    assert _arg_after(first, "--device") == "cuda:1"
    assert _arg_after(first, "--pilot-frame-batch-size") == "16"
    assert _arg_after(first, "--pilot-trace-batch-size") == "8"
    assert _arg_after(first, "--image-shard-start") == "0"
    assert _arg_after(first, "--image-shard-stop") == "50"
    assert _arg_after(first, "--checkpoint-path").endswith("epoch=147-val_bps_overall=0.5702.ckpt")
    assert _arg_after(first, "--dataset-configs").endswith("paper/fig4/upstream/dataset_configs/multi_basic_120_long.yaml")
    assert _arg_after(first, "--population-spec-dir").endswith(
        "outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints"
    )
    assert "--session-filter" not in first
    assert _arg_after(second, "--image-shard-start") == "50"
    assert _arg_after(second, "--image-shard-stop") == "100"
    assert "--skip-benchmark" in first

    serialized_commands = json.dumps([cmd["argv"] for cmd in manifest["commands"]])
    assert "/home/declan/VisionCore/" not in serialized_commands
    assert manifest["implementation_boundary"]["scorer_runner_status"] == "in_visioncoremain"
    assert manifest["implementation_boundary"]["stabilized_baseline_runner_status"] == "in_visioncoremain"
    merge = next(cmd for cmd in manifest["commands"] if cmd["stage"] == "merge_shards")
    assert merge["argv"][3].endswith("paper/fig4/upstream/merge_backimage_real_trace_ssi_matrix_shards.py")

    checks = {item["label"]: item for item in manifest["inputs"]}
    assert checks["direct_matrix_source_csv"]["expected_sha256"] == (
        "ac2364e22ede162940a9ba7de5c8ab2c2ef2ea9c9985d851e302769d17c12567"
    )
    assert checks["unit_tuning_csv"]["expected_sha256"] == (
        "7a506b617ccbda563cab1e7f10173f9015448f88ce71a1abec7b05dc8aaa92f2"
    )
    assert checks["window_features_csv"]["expected_sha256"] == (
        "e8e2fa28c39d4d0222502bbe73fc221210260212fbed25bdc6c2e6c6217f73ba"
    )
    assert checks["model_checkpoint"]["expected_sha256"] == (
        "55d084aa0beb7d65614aecb9122edf7ad49c5799d370dbbd5dcf60b815c62de3"
    )
    assert checks["dataset_configs_main"]["expected_sha256"] == (
        "c42906b90c340d64d35247baaa6b715e452d043dcff6439e02147aed6b7322d8"
    )
    assert checks["rr100_population_spec_json"]["expected_sha256"] == (
        "d599fb0718faa363520a91b8f0819edafbff74ec501899b590e4061fef557f08"
    )
    assert checks["mcfarland_outputs"]["required_for"] == ["score_shard", "stabilized_baseline"]


def test_smoke_plan_is_tiny_and_schema_compatible(tmp_path):
    manifest = _run_plan(tmp_path, "smoke")

    score = next(cmd for cmd in manifest["commands"] if cmd["stage"] == "score_shard")
    argv = score["argv"]
    assert manifest["selected_shards"] == ["images_000_001"]
    assert _arg_after(argv, "--n-images") == "1"
    assert _arg_after(argv, "--n-traces") == "2"
    assert _arg_after(argv, "--min-strong-contour-images") == "0"
    assert _arg_after(argv, "--min-microsaccade-traces") == "0"
    assert _arg_after(argv, "--device") == "cpu"
    assert _arg_after(argv, "--image-shard-stop") == "1"
    assert _arg_after(argv, "--session-filter") == "Allen_2022-02-16"
    assert manifest["profile_parameters"]["session_filter"] == "Allen_2022-02-16"


def test_session_filter_keeps_named_source_sessions():
    rows = pd.DataFrame(
        {
            "session": ["Allen_2022-02-16", "Allen_2022-03-04", "Allen_2022-02-16"],
            "value": [1, 2, 3],
        }
    )

    assert parse_session_filter("Allen_2022-02-16, Allen_2022-03-04") == [
        "Allen_2022-02-16",
        "Allen_2022-03-04",
    ]
    filtered = filter_source_rows(rows, ["Allen_2022-02-16"])

    assert filtered["value"].tolist() == [1, 3]


def test_trace_metric_helpers_preserve_historical_trace_bank_contract():
    trace = np.asarray([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]], dtype=np.float32)

    assert trace_hash(trace) == "3f981f84714f37bdfb67"
    assert speed_threshold_mad(trace, dt=1.0, z=2.0) == 2.0 + 2.0 * 1.4826

    metrics = trace_scale_metrics(trace, dt=0.5, prefix="rendered_")
    assert metrics["rendered_path_length_arcmin"] == 360.0
    assert np.isclose(metrics["rendered_rms_radius_arcmin"], np.sqrt(5.25) * 60.0)
    assert metrics["rendered_duration_s"] == 2.0
    assert "rendered_position_high_freq_power_fraction_15_60hz" in metrics


def test_checkpoint_defaults_prefer_env_then_staged(monkeypatch, tmp_path):
    modules = (real_trace_launcher, real_trace_scorer, stabilized_baseline, source_asset_bundle)
    for module in modules:
        staged = tmp_path / f"{module.__name__}_staged.ckpt"
        env_override = tmp_path / f"{module.__name__}_env.ckpt"
        staged.write_bytes(b"staged\n")
        staged_attr = "STAGED_MODEL_CHECKPOINT_PATH"
        if not hasattr(module, staged_attr):
            staged_attr = "STAGED_CHECKPOINT_PATH"
        monkeypatch.setattr(module, staged_attr, staged)
        monkeypatch.delenv(module.CHECKPOINT_ENV, raising=False)

        assert module.default_checkpoint_path() == staged

        monkeypatch.setenv(module.CHECKPOINT_ENV, str(env_override))

        assert module.default_checkpoint_path() == env_override


def test_trace_bank_metadata_sampling_matches_recovered_pandas_contract():
    rows = pd.DataFrame(
        {
            "source_row": np.arange(20, dtype=int),
            "n_samples": np.repeat(40, 20),
            "session": ["Allen_2022-02-16"] * 20,
        }
    )

    sampled = trace_bank_metadata_builder.sample_source_rows(
        rows,
        n_source_windows=7,
        seed=20260716,
        n_timepoints=40,
    )
    expected = rows.sample(n=7, random_state=20260716, replace=False).sort_values("source_row")

    assert sampled["source_row"].tolist() == expected["source_row"].tolist()


def test_trace_bank_metadata_legacy_rows_keep_diagnostic_schema():
    item = {
        "source_row": 10,
        "session": "Allen_2022-02-16",
        "trial_idx": 3,
        "global_start": 20,
        "global_stop": 60,
        "source_window_global_start": 0,
        "source_window_global_stop": 128,
        "snippet_global_start": 20,
        "snippet_global_stop": 60,
        "snippet_n_samples": 40,
        "snippet_duration_s": 39.0 / 120.0,
        "source_window_n_samples": 128,
        "source_window_duration_s": 127.0 / 120.0,
        "mean_x_deg": 0.1,
        "mean_y_deg": -0.1,
        "observed_rms_deg": 0.1,
        "source_trace_observed_rms_deg": 0.1,
        "path_length_deg": 0.2,
        "duration_s": 39.0 / 120.0,
        "lag1_autocorr": 0.5,
        "covariance_shape": np.eye(2),
        "trace_cov_anisotropy": 0.0,
        "source_trace_cov_anisotropy": 0.0,
        "source_anisotropy": 0.0,
        "trace_bank_snippet_policy": "center_crop_native_n_timepoints",
        "trace": np.zeros((40, 2), dtype=np.float32),
    }

    row = trace_bank_metadata_builder.legacy_trace_bank_metadata_rows([item])[0]

    assert list(row) == trace_bank_metadata_builder.LEGACY_TRACE_BANK_METADATA_COLUMNS
    assert len(row) == 156
    assert row["observed_rms_arcmin"] == 6.0
    assert row["path_length_arcmin"] == 12.0
    assert "trace" not in row


def test_run_all_refuses_without_required_source_assets(tmp_path):
    plan_json = tmp_path / "blocked.json"
    missing_root = tmp_path / "missing_assets"
    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--profile",
            "production",
            "--plan-json",
            str(plan_json),
            "--no-print-commands",
            "--source-csv",
            str(missing_root / "source.csv"),
            "--unit-tuning-csv",
            str(missing_root / "unit_tuning.csv"),
            "--checkpoint-path",
            str(missing_root / "checkpoint.ckpt"),
            "--population-spec-dir",
            str(missing_root / "population_spec_dir"),
            "--mcfarland-outputs",
            str(missing_root / "mcfarland_outputs_mono.pkl"),
            "--run-all",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Missing required source/model asset" in result.stderr
    assert "source CSV" in result.stderr
    assert "RR100 population JSON" in result.stderr
    assert "McFarland outputs" in result.stderr
    assert plan_json.exists()


def test_scorer_entrypoints_have_lightweight_help():
    for script in (
        MATRIX_RUNNER,
        BASELINE_RUNNER,
        STAGER,
        SOURCE_BUNDLE,
        TRACE_BANK_METADATA_BUILDER,
        COMPARATOR,
        AUDITOR,
    ):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "usage:" in result.stdout


class _FakeScorer:
    n_units = 3

    def score_traces_for_patch(
        self,
        patch,
        traces,
        *,
        trace_batch_size,
        frame_batch_size,
        n_timepoints,
        bin_seconds,
    ):
        assert patch.shape == (4, 4)
        assert trace_batch_size == 2
        assert frame_batch_size == 5
        assert n_timepoints == 4
        assert bin_seconds == 0.25
        base = np.arange(len(traces) * self.n_units, dtype=np.float32).reshape(len(traces), self.n_units)
        return base, base + 100.0, base + 200.0, np.arange(len(traces), dtype=np.float32) + 10.0


def test_score_matrix_writes_schema_with_fake_scorer(tmp_path):
    image_rows = pd.DataFrame(
        {
            "image_index": [0, 1],
            "source_row": [10, 11],
            "session": ["Allen_2022-02-16", "Allen_2022-02-16"],
            "trial_idx": [3, 4],
            "image_patch_rms_contrast": [0.5, 0.8],
        }
    )
    trace_items = [
        {
            "trace": np.zeros((4, 2), dtype=np.float32),
            "source_row": 20,
            "session": "Allen_2022-02-16",
            "trial_idx": 7,
            "rendered_path_length_arcmin": 1.0,
        },
        {
            "trace": np.ones((4, 2), dtype=np.float32),
            "source_row": 21,
            "session": "Allen_2022-02-16",
            "trial_idx": 8,
            "rendered_path_length_arcmin": 2.0,
        },
    ]

    def fake_patch_loader(row, *, canvas_cache, patch_size_px):
        return np.full((4, 4), float(row["image_index"]), dtype=np.float32), {"patch_size_px": int(patch_size_px)}

    timing = score_matrix(
        scorer=_FakeScorer(),
        image_rows=image_rows,
        trace_items=trace_items,
        frame_batch_size=5,
        trace_batch_size=2,
        n_timepoints=4,
        bin_seconds=0.25,
        patch_size_px=4,
        write_outputs=True,
        out_dir=tmp_path,
        patch_loader=fake_patch_loader,
    )

    assert timing["n_movies"] == 4
    assert np.load(tmp_path / "ssi_matrix.npy").shape == (4, 3)
    assert np.load(tmp_path / "expected_spikes_matrix.npy").shape == (4, 3)
    assert np.load(tmp_path / "population_ssi.npy").tolist() == [10.0, 11.0, 10.0, 11.0]
    movies = pd.read_csv(tmp_path / "movie_feature_table.csv")
    assert movies["movie_index"].tolist() == [0, 1, 2, 3]
    assert movies["image_source_row"].tolist() == [10, 10, 11, 11]


def test_replay_trace_loader_and_movie_stride_preserve_reference_indices(tmp_path):
    trace_table = pd.DataFrame(
        {
            "source_row": [20, 21],
            "session": ["Allen_2022-02-16", "Allen_2022-02-16"],
            "trial_idx": [7, 8],
        }
    )
    trace_xy = np.zeros((2, 4, 2), dtype=np.float32)
    traces = trace_items_from_table_and_array(trace_table, trace_xy, n_timepoints=4)
    image_rows = pd.DataFrame(
        {
            "image_index": [3],
            "source_row": [10],
            "session": ["Allen_2022-02-16"],
            "trial_idx": [3],
        }
    )

    def fake_patch_loader(row, *, canvas_cache, patch_size_px):
        return np.zeros((4, 4), dtype=np.float32), {"patch_size_px": int(patch_size_px)}

    score_matrix(
        scorer=_FakeScorer(),
        image_rows=image_rows,
        trace_items=traces,
        frame_batch_size=5,
        trace_batch_size=2,
        n_timepoints=4,
        bin_seconds=0.25,
        patch_size_px=4,
        write_outputs=True,
        out_dir=tmp_path,
        patch_loader=fake_patch_loader,
        trace_index_offset=10,
        movie_index_stride=1000,
    )

    movies = pd.read_csv(tmp_path / "movie_feature_table.csv")
    assert movies["movie_index"].tolist() == [3010, 3011]
    assert movies["trace_index"].tolist() == [10, 11]


def test_stage_real_trace_source_assets_symlinks_manifest(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    manifest = tmp_path / "manifest.json"
    for rel in SOURCE_ASSET_RELS:
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"dummy {rel}\n".encode("utf-8"))

    result = subprocess.run(
        [
            sys.executable,
            str(STAGER),
            str(source),
            "--target-root",
            str(target),
            "--manifest",
            str(manifest),
            "--apply",
            "--link-mode",
            "symlink",
            "--no-verify-hashes",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    statuses = {row["key"]: row["status"] for row in payload["assets"]}
    assert set(statuses.values()) == {"ok"}
    staged_pickle = target / "outputs/artifacts/mcfarland/mcfarland_outputs_mono.pkl"
    alias = target / "scripts/mcfarland_outputs_mono.pkl"
    assert staged_pickle.is_symlink()
    assert alias.is_symlink()
    assert alias.resolve() == staged_pickle.resolve()


def test_build_real_trace_source_asset_bundle_packages_portable_assets(tmp_path):
    source = tmp_path / "source"
    out_dir = tmp_path / "handoff"
    external_pickle = tmp_path / "external_mcfarland_outputs_mono.pkl"
    external_pickle.write_bytes(b"dummy external mcfarland payload\n")
    for rel in SOURCE_ASSET_RELS:
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("mcfarland_outputs_mono.pkl"):
            path.symlink_to(external_pickle)
        else:
            path.write_bytes(f"dummy {rel}\n".encode("utf-8"))

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_BUNDLE),
            "--source-root",
            str(source),
            "--out-dir",
            str(out_dir),
            "--name",
            "fig4_test_assets",
            "--no-verify-hashes",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    tar_path = out_dir / "fig4_test_assets.tar.gz"
    manifest_path = out_dir / "fig4_test_assets_manifest.json"
    checksum_path = out_dir / "fig4_test_assets_checksums.txt"
    assert tar_path.exists()
    assert manifest_path.exists()
    assert checksum_path.exists()

    with tarfile.open(tar_path, "r:gz") as tar:
        members = {member.name: member for member in tar.getmembers()}
        for rel in SOURCE_ASSET_RELS:
            assert rel in members
            assert members[rel].isfile()
        alias = members["scripts/mcfarland_outputs_mono.pkl"]
        assert alias.issym()
        assert alias.linkname == "../outputs/artifacts/mcfarland/mcfarland_outputs_mono.pkl"
        bundled_manifest = "outputs/figures/fig4/provenance/real_trace_source_asset_bundle_manifest.json"
        assert bundled_manifest in members
        handle = tar.extractfile(bundled_manifest)
        assert handle is not None
        payload = json.loads(handle.read().decode("utf-8"))

    assert payload["analysis"] == "fig4_real_trace_source_asset_bundle"
    assert payload["include_checkpoint"] is False
    assert [row["key"] for row in payload["assets"]] == [
        "direct_matrix_source_csv",
        "window_features_csv",
        "unit_tuning_csv",
        "rr100_population_spec_json",
        "rr100_population_spec_npz",
        "mcfarland_outputs_mono",
    ]
    assert payload["assets"][-1]["alias"]["archive_path"] == "scripts/mcfarland_outputs_mono.pkl"


def test_compare_real_trace_matrix_outputs_aligns_by_movie_index(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    reference_movies = pd.DataFrame({"movie_index": [0, 1, 1000, 1001]})
    candidate_movies = pd.DataFrame({"movie_index": [1000, 1001]})
    reference_movies.to_csv(reference / "movie_feature_table.csv", index=False)
    candidate_movies.to_csv(candidate / "movie_feature_table.csv", index=False)
    pd.DataFrame({"image_index": [0, 1], "image_source_row": [10, 11]}).to_csv(
        reference / "image_feature_table.csv", index=False
    )
    pd.DataFrame({"image_index": [1], "image_source_row": [11]}).to_csv(
        candidate / "image_feature_table.csv", index=False
    )
    pd.DataFrame({"trace_bank_index": [0, 1, 2], "trace_hash": ["a", "b", "c"]}).to_csv(
        reference / "trace_feature_table.csv", index=False
    )
    pd.DataFrame({"trace_bank_index": [1, 2], "trace_hash": ["b", "c"]}).to_csv(
        candidate / "trace_feature_table.csv", index=False
    )
    pd.DataFrame({"unit_index": [0, 1], "unit_label": ["u0", "u1"]}).to_csv(
        reference / "unit_feature_table.csv", index=False
    )
    pd.DataFrame({"unit_index": [0, 1], "unit_label": ["u0", "u1"]}).to_csv(
        candidate / "unit_feature_table.csv", index=False
    )
    base = np.arange(8, dtype=np.float32).reshape(4, 2)
    for name in ("ssi_matrix.npy", "expected_spikes_matrix.npy", "mean_rate_matrix.npy"):
        np.save(reference / name, base)
        np.save(candidate / name, base[2:4])
    np.save(reference / "population_ssi.npy", np.arange(4, dtype=np.float32))
    np.save(candidate / "population_ssi.npy", np.arange(4, dtype=np.float32)[2:4])

    result = subprocess.run(
        [
            sys.executable,
            str(COMPARATOR),
            "--candidate-dir",
            str(candidate),
            "--reference-dir",
            str(reference),
            "--atol",
            "0",
            "--rtol",
            "0",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[pass] ssi_matrix.npy" in result.stdout
    assert "[pass] trace_feature_table.csv" in result.stdout


def test_replay_audit_plans_scorer_and_comparator(tmp_path):
    plan_json = tmp_path / "audit_plan.json"
    out_dir = tmp_path / "audit_out"
    reference = tmp_path / "reference"
    result = subprocess.run(
        [
            sys.executable,
            str(AUDITOR),
            "--reference-dir",
            str(reference),
            "--out-dir",
            str(out_dir),
            "--plan-json",
            str(plan_json),
            "--image-start",
            "2",
            "--image-stop",
            "4",
            "--trace-start",
            "10",
            "--trace-stop",
            "15",
            "--no-print-commands",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(plan_json.read_text(encoding="utf-8"))
    assert payload["image_bounds"] == [2, 4]
    assert payload["trace_bounds"] == [10, 15]
    assert [cmd["label"] for cmd in payload["commands"]] == ["replay_score", "compare"]
    scorer_argv = payload["commands"][0]["argv"]
    assert _arg_after(scorer_argv, "--n-images") == "2"
    assert _arg_after(scorer_argv, "--n-traces") == "5"
    assert _arg_after(scorer_argv, "--image-shard-start") == "2"
    assert _arg_after(scorer_argv, "--trace-shard-start") == "10"


def _write_shard(path: Path, *, movie_indices: list[int], matrix_offset: float) -> None:
    path.mkdir(parents=True)
    image_table = pd.DataFrame({"image_index": [0, 1], "image_feature": [0.2, 0.7]})
    trace_table = pd.DataFrame({"trace_index": [0, 1, 2], "trace_feature": [1.0, 2.0, 3.0]})
    unit_table = pd.DataFrame({"unit_index": [0, 1], "unit_label": ["u000", "u001"]})
    movie_table = pd.DataFrame(
        {
            "movie_index": movie_indices,
            "matrix_row_index": list(range(len(movie_indices))),
            "image_index": [idx // 3 for idx in movie_indices],
            "trace_index": [idx % 3 for idx in movie_indices],
        }
    )
    image_table.to_csv(path / "image_feature_table.csv", index=False)
    trace_table.to_csv(path / "trace_feature_table.csv", index=False)
    unit_table.to_csv(path / "unit_feature_table.csv", index=False)
    movie_table.to_csv(path / "movie_feature_table.csv", index=False)
    (path / "summary.json").write_text('{"analysis": "synthetic"}\n', encoding="utf-8")
    np.save(path / "trace_xy.npy", np.arange(18, dtype=np.float32).reshape(3, 3, 2))
    base = np.arange(len(movie_indices) * 2, dtype=np.float32).reshape(len(movie_indices), 2)
    for name in ("ssi_matrix.npy", "expected_spikes_matrix.npy", "mean_rate_matrix.npy"):
        np.save(path / name, base + matrix_offset)
    np.save(path / "population_ssi.npy", np.arange(len(movie_indices), dtype=np.float32) + matrix_offset)


def test_merge_real_trace_shards_reconstructs_movie_index_order(tmp_path):
    shard_a = tmp_path / "images_000_001"
    shard_b = tmp_path / "images_001_002"
    out_dir = tmp_path / "merged"
    _write_shard(shard_a, movie_indices=[0, 1, 2], matrix_offset=10.0)
    _write_shard(shard_b, movie_indices=[3, 4, 5], matrix_offset=20.0)

    result = subprocess.run(
        [sys.executable, str(MERGER), "--out-dir", str(out_dir), str(shard_a), str(shard_b)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    ssi = np.load(out_dir / "ssi_matrix.npy")
    population = np.load(out_dir / "population_ssi.npy")
    movies = pd.read_csv(out_dir / "movie_feature_table.csv")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    assert ssi.shape == (6, 2)
    np.testing.assert_array_equal(ssi[0], np.array([10.0, 11.0], dtype=np.float32))
    np.testing.assert_array_equal(ssi[3], np.array([20.0, 21.0], dtype=np.float32))
    np.testing.assert_array_equal(population, np.array([10.0, 11.0, 12.0, 20.0, 21.0, 22.0], dtype=np.float32))
    assert movies["movie_index"].tolist() == [0, 1, 2, 3, 4, 5]
    assert summary["n_images"] == 2
    assert summary["n_traces"] == 3
    assert summary["n_movies"] == 6
