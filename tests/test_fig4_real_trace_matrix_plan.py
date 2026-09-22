from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "paper/fig4/upstream"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

import run_real_trace_matrix as launcher
import score_real_trace_stabilized_baseline as stabilized_baseline
from real_trace_matrix.core import (
    score_matrix,
    speed_threshold_mad,
    trace_hash,
    trace_items_from_table_and_array,
    trace_scale_metrics,
)
from score_real_trace_matrix import filter_source_rows, parse_session_filter


MERGER = UPSTREAM / "merge_backimage_real_trace_ssi_matrix_shards.py"
SCORER = UPSTREAM / "score_real_trace_matrix.py"
BASELINE = UPSTREAM / "score_real_trace_stabilized_baseline.py"


def _write_model_contract(tmp_path: Path, *, output_rate: int = 240) -> Path:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        "sampling: {source_rate: 240, target_rate: 240}\n"
        f"supervision: {{target_rate: {output_rate}, phase: 0}}\n"
        "keys_lags:\n  stim: ["
        + ", ".join(map(str, range(60)))
        + "]\n",
        encoding="utf-8",
    )
    spec = tmp_path / "model.yaml"
    spec.write_text(
        "label: test-model\n"
        "checkpoint:\n"
        f"  path: {checkpoint}\n"
        f"  sha256: {launcher.sha256(checkpoint)}\n"
        "training:\n"
        "  datasets:\n"
        "    descriptive_all_gratings:\n"
        f"      path: {dataset}\n"
        f"      sha256: {launcher.sha256(dataset)}\n",
        encoding="utf-8",
    )
    return spec


def _write_population(tmp_path: Path, *, pooling_mode: str = "exact_identity") -> tuple[Path, str]:
    directory = tmp_path / "population"
    directory.mkdir()
    version = "all_exact_v1"
    (directory / f"population_spec_{version}.json").write_text(
        json.dumps({"version": version, "pooling_mode": pooling_mode}) + "\n",
        encoding="utf-8",
    )
    np.savez(directory / f"population_spec_{version}.npz", membership=np.eye(2))
    return directory, version


def _write_replay_bank(tmp_path: Path, *, n_images: int = 40, n_traces: int = 200) -> Path:
    directory = tmp_path / "replay"
    directory.mkdir()
    pd.DataFrame({"image_index": np.arange(n_images)}).to_csv(
        directory / "image_feature_table.csv", index=False
    )
    pd.DataFrame({"trace_index": np.arange(n_traces)}).to_csv(
        directory / "trace_feature_table.csv", index=False
    )
    np.save(directory / "trace_xy.npy", np.zeros((n_traces, 60, 2), dtype=np.float32))
    (directory / "trace_provenance.json").write_text("{}\n", encoding="utf-8")
    return directory


def test_model_contract_is_native_240_and_digest_bound(tmp_path: Path) -> None:
    contract = launcher.load_model_contract(_write_model_contract(tmp_path))
    assert contract["n_timepoints"] == 60
    assert contract["bin_seconds"] == 1 / 240
    assert contract["checkpoint_sha256"] == launcher.sha256(contract["checkpoint"])

    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(ValueError, match="240->240"):
        launcher.load_model_contract(_write_model_contract(bad, output_rate=120))


def test_population_contract_refuses_pooling(tmp_path: Path) -> None:
    exact_dir, version = _write_population(tmp_path)
    assert launcher.population_contract(exact_dir, version)["npz"].is_file()
    pooled = tmp_path / "pooled"
    pooled.mkdir()
    pooled_dir, pooled_version = _write_population(pooled, pooling_mode="cluster_mean")
    with pytest.raises(ValueError, match="exact_identity"):
        launcher.population_contract(pooled_dir, pooled_version)


def test_release_profile_builds_four_explicit_exact_unit_shards(tmp_path: Path) -> None:
    spec = _write_model_contract(tmp_path)
    population_dir, population_version = _write_population(tmp_path)
    replay = _write_replay_bank(tmp_path)
    unit_table = tmp_path / "units.csv"
    unit_table.write_text("unit_index\n0\n", encoding="utf-8")
    mcfarland = tmp_path / "mcfarland.pkl"
    mcfarland.write_bytes(b"metadata")
    args = Namespace(
        profile="release",
        n_images=None,
        n_traces=None,
        image_shard_size=None,
        device=None,
        frame_batch_size=None,
        trace_batch_size=None,
        population_version=population_version,
        population_spec_dir=population_dir,
        replay_matrix_dir=replay,
        unit_table=unit_table,
        mcfarland_outputs=mcfarland,
        force=False,
    )
    profile = launcher.selected_profile(args)
    launcher.validate_replay_bank(replay, n_images=40, n_traces=200)
    commands = launcher.build_commands(
        contract=launcher.load_model_contract(spec),
        args=args,
        profile=profile,
        output_root=tmp_path / "output",
    )
    assert profile["n_images"] == 40
    assert profile["n_traces"] == 200
    assert len(commands["score"]) == 4
    assert "--population-version" in commands["score"][0]
    assert "--rr100-version" not in commands["score"][0]
    assert commands["score"][0][commands["score"][0].index("--n-timepoints") + 1] == "60"
    assert commands["score"][-1][commands["score"][-1].index("--image-shard-stop") + 1] == "40"


def test_replay_bank_requires_filtered_native_240_shape(tmp_path: Path) -> None:
    replay = _write_replay_bank(tmp_path, n_images=2, n_traces=3)
    launcher.validate_replay_bank(replay, n_images=2, n_traces=3)
    np.save(replay / "trace_xy.npy", np.zeros((3, 40, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="60, 2"):
        launcher.validate_replay_bank(replay, n_images=2, n_traces=3)


def test_execution_requires_yatesfv() -> None:
    launcher.require_execution_environment(execute=False, environment=None)
    launcher.require_execution_environment(execute=True, environment="yatesfv")
    with pytest.raises(RuntimeError, match="yatesfv"):
        launcher.require_execution_environment(execute=True, environment="base")


def test_session_filter_keeps_named_source_sessions() -> None:
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
    assert filter_source_rows(rows, ["Allen_2022-02-16"])["value"].tolist() == [1, 3]


def test_trace_metric_helpers_preserve_units() -> None:
    trace = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]], dtype=np.float32
    )
    assert trace_hash(trace) == "3f981f84714f37bdfb67"
    assert speed_threshold_mad(trace, dt=1.0, z=2.0) == 2.0 + 2.0 * 1.4826
    metrics = trace_scale_metrics(trace, dt=0.5, prefix="rendered_")
    assert metrics["rendered_path_length_arcmin"] == 360.0
    assert np.isclose(metrics["rendered_rms_radius_arcmin"], np.sqrt(5.25) * 60.0)


def test_scorer_entrypoints_have_lightweight_help() -> None:
    for script in (SCORER, BASELINE):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


class _FakeScorer:
    n_units = 3

    def score_traces_for_patch(
        self, patch, traces, *, trace_batch_size, frame_batch_size, n_timepoints, bin_seconds
    ):
        assert patch.shape == (4, 4)
        assert (trace_batch_size, frame_batch_size, n_timepoints, bin_seconds) == (2, 5, 4, 0.25)
        base = np.arange(len(traces) * self.n_units, dtype=np.float32).reshape(
            len(traces), self.n_units
        )
        return base, base + 100.0, base + 200.0, np.arange(len(traces), dtype=np.float32) + 10.0


def test_score_matrix_writes_schema_with_fake_scorer(tmp_path: Path) -> None:
    image_rows = pd.DataFrame(
        {
            "image_index": [0, 1],
            "source_row": [10, 11],
            "session": ["Allen_2022-02-16"] * 2,
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
        return np.full((4, 4), float(row["image_index"]), dtype=np.float32), {
            "patch_size_px": int(patch_size_px)
        }

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
    assert pd.read_csv(tmp_path / "movie_feature_table.csv")["movie_index"].tolist() == [0, 1, 2, 3]


def test_replay_trace_loader_preserves_reference_indices(tmp_path: Path) -> None:
    trace_table = pd.DataFrame(
        {
            "source_row": [20, 21],
            "session": ["Allen_2022-02-16"] * 2,
            "trial_idx": [7, 8],
        }
    )
    traces = trace_items_from_table_and_array(
        trace_table, np.zeros((2, 4, 2), dtype=np.float32), n_timepoints=4
    )
    assert [item["source_row"] for item in traces] == [20, 21]


def _write_shard(path: Path, *, movie_indices: list[int], matrix_offset: float) -> None:
    path.mkdir(parents=True)
    pd.DataFrame({"image_index": [0, 1], "image_feature": [0.2, 0.7]}).to_csv(
        path / "image_feature_table.csv", index=False
    )
    pd.DataFrame({"trace_index": [0, 1, 2], "trace_feature": [1.0, 2.0, 3.0]}).to_csv(
        path / "trace_feature_table.csv", index=False
    )
    pd.DataFrame({"unit_index": [0, 1], "unit_label": ["u000", "u001"]}).to_csv(
        path / "unit_feature_table.csv", index=False
    )
    pd.DataFrame(
        {
            "movie_index": movie_indices,
            "matrix_row_index": range(len(movie_indices)),
            "image_index": [index // 3 for index in movie_indices],
            "trace_index": [index % 3 for index in movie_indices],
        }
    ).to_csv(path / "movie_feature_table.csv", index=False)
    summary = {
        "analysis": "synthetic",
        "population_version": "synthetic-exact-v1",
        "bin_seconds": 1 / 240,
        "n_timepoints": 60,
        "patch_size_px": 35,
        "source_csv": None,
        "unit_tuning_csv": "synthetic_tuning.csv",
        "trace_time_contract": {
            "source_trace_rate_hz": 240,
            "source_trace_samples": 60,
            "model_output_rate_hz": 240,
            "scored_trace_samples": 60,
        },
        "model_provenance": {
            "model": {
                "checkpoint_sha256": "checkpoint-hash",
                "dataset_configs_sha256": "dataset-hash",
            },
            "population_spec_json_sha256": "population-json-hash",
            "population_spec_npz_sha256": "population-npz-hash",
            "population_contract": {"pooling_mode": "exact_identity"},
            "stimulus": {
                "model_history_frames": 60,
                "model_input_rate_hz": 240,
                "model_output_rate_hz": 240,
                "supervision_phase": 0,
            },
        },
    }
    (path / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    np.save(path / "trace_xy.npy", np.arange(360, dtype=np.float32).reshape(3, 60, 2))
    base = np.arange(len(movie_indices) * 2, dtype=np.float32).reshape(len(movie_indices), 2)
    for name in ("ssi_matrix.npy", "expected_spikes_matrix.npy", "mean_rate_matrix.npy"):
        np.save(path / name, base + matrix_offset)
    np.save(path / "population_ssi.npy", np.arange(len(movie_indices), dtype=np.float32) + matrix_offset)


def test_merge_real_trace_shards_reconstructs_movie_order(tmp_path: Path) -> None:
    shard_a = tmp_path / "images_000_001"
    shard_b = tmp_path / "images_001_002"
    out_dir = tmp_path / "merged"
    _write_shard(shard_a, movie_indices=[0, 1, 2], matrix_offset=10.0)
    _write_shard(shard_b, movie_indices=[3, 4, 5], matrix_offset=20.0)
    result = subprocess.run(
        [sys.executable, str(MERGER), "--out-dir", str(out_dir), str(shard_a), str(shard_b)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert np.load(out_dir / "ssi_matrix.npy").shape == (6, 2)
    assert pd.read_csv(out_dir / "movie_feature_table.csv")["movie_index"].tolist() == list(range(6))
    assert json.loads((out_dir / "summary.json").read_text())["n_movies"] == 6


def test_stabilized_baseline_uses_only_scored_images_for_a_shard(tmp_path: Path) -> None:
    full = tmp_path / "image_feature_table.csv"
    scored = tmp_path / "scored_image_feature_table.csv"
    full.write_text("image_index\n0\n1\n2\n", encoding="utf-8")
    assert stabilized_baseline.selected_image_table_path(tmp_path) == full
    scored.write_text("image_index\n1\n", encoding="utf-8")
    assert stabilized_baseline.selected_image_table_path(tmp_path) == scored
