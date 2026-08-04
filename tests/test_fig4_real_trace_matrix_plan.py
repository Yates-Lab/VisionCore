from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "paper" / "fig4" / "upstream" / "run_real_trace_matrix.py"
MERGER = ROOT / "paper" / "fig4" / "upstream" / "merge_backimage_real_trace_ssi_matrix_shards.py"
RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_"
    "medoidPosthocminRepcomplete0p45_movieMedoid"
)


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
    assert _arg_after(second, "--image-shard-start") == "50"
    assert _arg_after(second, "--image-shard-stop") == "100"
    assert "--skip-benchmark" in first

    serialized_commands = json.dumps([cmd["argv"] for cmd in manifest["commands"]])
    assert "/home/declan/VisionCore/" not in serialized_commands
    merge = next(cmd for cmd in manifest["commands"] if cmd["stage"] == "merge_shards")
    assert merge["argv"][3].endswith("paper/fig4/upstream/merge_backimage_real_trace_ssi_matrix_shards.py")

    checks = {item["label"]: item for item in manifest["inputs"]}
    assert checks["model_checkpoint"]["expected_sha256"] == (
        "55d084aa0beb7d65614aecb9122edf7ad49c5799d370dbbd5dcf60b815c62de3"
    )
    assert checks["dataset_configs_main"]["expected_sha256"] == (
        "c42906b90c340d64d35247baaa6b715e452d043dcff6439e02147aed6b7322d8"
    )
    assert checks["rr100_population_spec_json"]["expected_sha256"] == (
        "d599fb0718faa363520a91b8f0819edafbff74ec501899b590e4061fef557f08"
    )


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


def test_run_all_refuses_without_explicit_scorer_runner(tmp_path):
    plan_json = tmp_path / "blocked.json"
    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--profile",
            "production",
            "--plan-json",
            str(plan_json),
            "--no-print-commands",
            "--run-all",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "FIG4_REAL_TRACE_MATRIX_RUNNER" in result.stderr
    assert "FIG4_STABILIZED_BASELINE_RUNNER" in result.stderr
    assert plan_json.exists()


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
