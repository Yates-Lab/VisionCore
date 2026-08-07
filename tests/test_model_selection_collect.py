"""Behaviour of `paper/model_selection/collect.py` and `stability.py`.

The sweep is read once, at the end, across fifteen arms run over days. Three
things have to hold for that reading to mean anything, and all three are cheap
to pin here because none of them needs a GPU or a checkpoint:

1. A run produced under a different protocol must not enter the pool.
2. A run that has not been evaluated must report an *absent* metric, never a
   zero -- a zero in a BPS column is a plausible-looking bad model.
3. A delta must be read against the replicate spread, so the replicate group
   has to be found correctly and the spread arithmetic has to be right.

Synthetic run directories throughout: a manifest, some empty `.ckpt` files whose
names carry the metric, and optionally an `evaluation.json`. That is exactly the
on-disk surface `collect.py` reads.
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MODEL_SELECTION = HERE.parent / "paper" / "model_selection"
if str(MODEL_SELECTION) not in sys.path:
    sys.path.insert(0, str(MODEL_SELECTION))

import collect  # noqa: E402
import launch  # noqa: E402
import protocol  # noqa: E402
import stability  # noqa: E402


BASE_SPEC = {
    "config": "multi_120_long_split3.yaml",
    "model_config": "experiments/model_configs/learned_resnet_concat_convgru_gaussian.yaml",
    "width": 1.0,
    "batch_size": 256,
    "lr": 1e-3,
    "core_lr_scale": 1.0,
    "wd": 1e-5,
    "homogeneous": False,
    "samples": 8_000_000,
    "seed": 101,
    "note": "synthetic",
    "accumulate": 4,
    "effective_batch": 1024,
    "max_epochs": 61,
    "samples_actual": 7_995_392,
}


def make_run(root, name, *, val_bps=0.60, epochs=(59,), spec_over=None,
             evaluation=None, protocol_hash=None, manifest=True,
             eval_protocol_hash=None):
    """Write a synthetic run directory and return its path."""
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    spec = dict(BASE_SPEC)
    spec.update(spec_over or {})
    spec["name"] = name

    if manifest:
        (run_dir / "manifest.json").write_text(json.dumps({
            "run": name,
            "protocol_hash": protocol_hash or protocol.PROTOCOL_HASH,
            "protocol": protocol.protocol_dict(),
            "spec": spec,
            "command": ["uv", "run", "python", "training/train_multidataset.py"],
            "launched": "2026-08-04T12:00:00",
        }, indent=2))

    for i, epoch in enumerate(epochs):
        # Only the last epoch carries the winning score, so `best_checkpoint`
        # has something to choose between.
        score = val_bps if i == len(epochs) - 1 else val_bps - 0.01 * (len(epochs) - i)
        (run_dir / f"epoch={epoch:02d}-val_bps_overall={score:.4f}.ckpt").touch()
    if epochs:
        (run_dir / "last.ckpt").touch()

    if evaluation is not None:
        payload = dict(evaluation)
        payload.setdefault("run", name)
        payload["protocol_hash"] = eval_protocol_hash or protocol.PROTOCOL_HASH
        (run_dir / "evaluation.json").write_text(json.dumps(payload, indent=2))
    return run_dir


def evaluation_payload(test_bps=0.61, ccnorm=0.59, r2=0.030, n_units=892):
    return {
        "test_split": {"bps_overall": test_bps, "bps_by_session": {}},
        "fixrsvp": {
            "n_sessions": 17,
            "n_units": n_units,
            "metrics": {
                "ccnorm": {"median": ccnorm, "n": n_units},
                "single_trial_r2": {"median": r2, "n": n_units},
                "bps": {"median": 0.13, "n": n_units},
            },
            "by_session": {},
        },
    }


# ---------------------------------------------------------------------------
# Reading one run
# ---------------------------------------------------------------------------
def test_reads_val_bps_from_checkpoint_name(tmp_path):
    """Validation BPS exists only in the filename; nothing else records it."""
    make_run(tmp_path, "E1a", val_bps=0.5958, epochs=(47, 55, 59))
    row = collect.load_run(tmp_path / "E1a")

    assert row["val_bps"] == pytest.approx(0.5958)
    assert row["best_checkpoint"] == "epoch=59-val_bps_overall=0.5958.ckpt"
    assert row["n_checkpoints"] == 3          # last.ckpt is not selectable
    assert row["last_epoch"] == 59


def test_reads_evaluation_metrics(tmp_path):
    make_run(tmp_path, "E1a", evaluation=evaluation_payload(
        test_bps=0.6090, ccnorm=0.5893, r2=0.0303))
    row = collect.load_run(tmp_path / "E1a")

    assert row["evaluated"] is True
    assert row["test_bps"] == pytest.approx(0.6090)
    assert row["ccnorm"] == pytest.approx(0.5893)
    assert row["single_trial_r2"] == pytest.approx(0.0303)
    assert row["fixrsvp_units"] == 892


def test_unevaluated_run_reports_absent_not_zero(tmp_path):
    """A missing evaluation must not read as a model that scored zero."""
    make_run(tmp_path, "E1b", evaluation=None)
    row = collect.load_run(tmp_path / "E1b")

    assert row["evaluated"] is False
    assert row["test_bps"] is None
    assert row["ccnorm"] is None
    assert row["val_bps"] is not None      # training did happen


# ---------------------------------------------------------------------------
# The protocol gate
# ---------------------------------------------------------------------------
def test_manifest_from_another_protocol_raises(tmp_path):
    make_run(tmp_path, "OLD", protocol_hash="deadbeefcafe")
    with pytest.raises(ValueError, match="deadbeefcafe"):
        collect.load_run(tmp_path / "OLD")


def test_evaluation_from_another_protocol_raises(tmp_path):
    """The report can postdate the manifest, so it is checked in its own right."""
    make_run(tmp_path, "E1a", evaluation=evaluation_payload(),
             eval_protocol_hash="deadbeefcafe")
    with pytest.raises(ValueError, match="deadbeefcafe"):
        collect.load_run(tmp_path / "E1a")


def test_protocol_mismatch_propagates_out_of_load_runs(tmp_path):
    """Not downgraded to a skip: a mixed pool is the failure being prevented."""
    make_run(tmp_path, "E1a")
    make_run(tmp_path, "OLD", protocol_hash="deadbeefcafe")
    with pytest.raises(ValueError):
        collect.load_runs(tmp_path)


def test_unmanifested_directory_is_skipped_with_a_reason(tmp_path):
    """The smoke-test dirs live beside the arms; drop them, but say so."""
    make_run(tmp_path, "E1a")
    make_run(tmp_path, "SMOKE_A", manifest=False)

    rows, skipped = collect.load_runs(tmp_path)
    assert [r["run"] for r in rows] == ["E1a"]
    assert len(skipped) == 1
    assert skipped[0][0] == "SMOKE_A"
    assert "manifest.json" in skipped[0][1]


# ---------------------------------------------------------------------------
# Replicates and deltas
# ---------------------------------------------------------------------------
def test_signature_ignores_seed_so_replicates_group(tmp_path):
    a = collect.config_signature(dict(BASE_SPEC, seed=101))
    b = collect.config_signature(dict(BASE_SPEC, seed=102))
    c = collect.config_signature(dict(BASE_SPEC, seed=101, lr=3e-3))
    assert a == b
    assert a != c


def test_baseline_group_finds_all_replicates(tmp_path):
    make_run(tmp_path, "E1a", spec_over={"seed": 101})
    make_run(tmp_path, "E2b", spec_over={"seed": 102})
    make_run(tmp_path, "E3b", spec_over={"seed": 103})
    make_run(tmp_path, "E3c", spec_over={"lr": 3e-3})

    rows, _ = collect.load_runs(tmp_path)
    group = collect.baseline_group(rows, "E1a")
    assert sorted(r["run"] for r in group) == ["E1a", "E2b", "E3b"]


def test_deltas_are_against_the_replicate_mean(tmp_path):
    make_run(tmp_path, "E1a", val_bps=0.59, spec_over={"seed": 101})
    make_run(tmp_path, "E2b", val_bps=0.60, spec_over={"seed": 102})
    make_run(tmp_path, "E3b", val_bps=0.61, spec_over={"seed": 103})
    make_run(tmp_path, "E3c", val_bps=0.65, spec_over={"lr": 3e-3})

    rows, _ = collect.load_runs(tmp_path)
    ref, group = collect.attach_deltas(rows, "E1a")

    assert ref["val_bps"] == pytest.approx(0.60)      # mean, not E1a's 0.59
    arm = next(r for r in rows if r["run"] == "E3c")
    assert arm["d_val_bps"] == pytest.approx(0.05)
    assert arm["is_baseline_replicate"] is False


def test_no_baseline_means_no_deltas(tmp_path):
    make_run(tmp_path, "E3c", val_bps=0.65, spec_over={"lr": 3e-3})
    rows, _ = collect.load_runs(tmp_path)
    ref, group = collect.attach_deltas(rows, "E1a")

    assert group == []
    assert ref["val_bps"] is None
    assert rows[0]["d_val_bps"] is None


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------
def test_spread_of_three():
    f = stability.spread([0.59, 0.60, 0.62])
    assert f["n"] == 3
    assert f["mean"] == pytest.approx(0.6033, abs=1e-4)
    assert f["range"] == pytest.approx(0.03)
    assert f["sd"] == pytest.approx(0.01528, abs=1e-4)


def test_spread_of_one_has_no_floor():
    f = stability.spread([0.59])
    assert f["n"] == 1
    assert f["range"] is None
    assert f["sd"] is None


def test_spread_ignores_absent_values():
    assert stability.spread([0.59, None, 0.61])["n"] == 2


def test_verdict_thresholds():
    floor = {"n": 3, "mean": 0.60, "sd": 0.015, "range": 0.03}
    assert stability.verdict(0.05, floor) == "resolved"      # beyond the range
    assert stability.verdict(-0.05, floor) == "resolved"     # sign-blind
    assert stability.verdict(0.02, floor) == "marginal"      # inside range, > sd
    assert stability.verdict(0.005, floor) == "unresolved"
    assert stability.verdict(0.05, {"n": 1, "range": None}) == "n/a"
    assert stability.verdict(None, floor) == "n/a"


def test_analyse_excludes_replicates_from_the_arm_list(tmp_path):
    make_run(tmp_path, "E1a", val_bps=0.59, spec_over={"seed": 101})
    make_run(tmp_path, "E2b", val_bps=0.60, spec_over={"seed": 102})
    make_run(tmp_path, "E3b", val_bps=0.61, spec_over={"seed": 103})
    make_run(tmp_path, "E1b", val_bps=0.70, spec_over={"homogeneous": True})

    rows, _ = collect.load_runs(tmp_path)
    result = stability.analyse(rows, "E1a")

    assert [a["run"] for a in result["arms"]] == ["E1b"]
    assert result["floors"]["val_bps"]["range"] == pytest.approx(0.02)
    assert result["arms"][0]["verdict_val_bps"] == "resolved"


def test_missing_replicates_names_the_arms_still_to_run(tmp_path):
    """With only E1a in hand, the floor is not yet real -- say what is missing."""
    make_run(tmp_path, "E1a", spec_over={"seed": 101})
    rows, _ = collect.load_runs(tmp_path)
    assert stability.missing_replicates(rows, "E1a") == ["E2b", "E3b"]


# ---------------------------------------------------------------------------
# Status labelling
# ---------------------------------------------------------------------------
def test_status_allows_one_validation_interval_of_slack():
    """Checkpoints land only on validation, so a finished run trails the horizon."""
    assert collect.run_status(59, 61) == "done"
    assert collect.run_status(20, 61) == "partial"
    assert collect.run_status(None, 61) == "no ckpt"


# ---------------------------------------------------------------------------
# The final model is guarded
# ---------------------------------------------------------------------------
def test_final_model_refuses_to_build_while_unsettled():
    """Training the defaults and calling it 'final' is the failure being designed out."""
    assert launch.unsettled_final(), "FINAL_SETTINGS should start empty"
    with pytest.raises(SystemExit, match="not settled"):
        launch.resolve("FINAL")


def test_final_model_builds_once_settled(monkeypatch):
    monkeypatch.setitem(launch.FINAL_SETTINGS, "homogeneous", False)
    monkeypatch.setitem(launch.FINAL_SETTINGS, "effective_batch", 1024)
    monkeypatch.setitem(launch.FINAL_SETTINGS, "lr", 1e-3)
    monkeypatch.setitem(launch.FINAL_SETTINGS, "core_lr_scale", 1.0)
    monkeypatch.setitem(launch.FINAL_SETTINGS, "samples", 16_000_000)
    monkeypatch.setitem(launch.FINAL_SETTINGS, "width", 1.0)

    spec = launch.resolve("FINAL")
    assert spec["name"] == "FINAL"
    assert spec["max_epochs"] == 122          # 16M / (512 * 256)
    assert spec["accumulate"] == 4

    cmd, ckpt_dir = launch.build_command(spec, gpu=0)
    assert "--no-homogeneous_batches" in cmd
    assert "--no-early_stopping" in cmd
    assert ckpt_dir.name == "FINAL"


def test_final_model_uses_the_same_builder_as_the_arms(monkeypatch):
    """The flags must come from build_command, not from a copy in a shell script."""
    for key, value in {"homogeneous": False, "effective_batch": 1024,
                       "lr": 1e-3, "core_lr_scale": 1.0,
                       "samples": 8_000_000, "width": 1.0}.items():
        monkeypatch.setitem(launch.FINAL_SETTINGS, key, value)

    final_cmd, _ = launch.build_command(launch.resolve("FINAL"), gpu=0)
    arm_cmd, _ = launch.build_command(launch.resolve("E1a"), gpu=0)

    def flags(cmd):
        return {c for c in cmd if c.startswith("--")}

    assert flags(final_cmd) == flags(arm_cmd)


# ---------------------------------------------------------------------------
# Evaluation batching is pinned, not inherited
# ---------------------------------------------------------------------------
def test_test_split_is_scored_with_cross_session_batching(monkeypatch):
    """A homogeneous arm must not score its test split on a ~63% subsample.

    `ByDatasetBatchSampler` draws with replacement across batches, so iterating
    it covers only part of the split. Inheriting the arm's training batching
    would make a homogeneous arm's BPS and a cross-session arm's BPS describe
    different amounts of data while being differenced against each other.
    """
    import evaluate

    captured = {}

    class FakeDM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import training.pl_modules as pl_modules
    monkeypatch.setattr(pl_modules, "MultiDatasetDM", FakeDM)

    spec = dict(BASE_SPEC, homogeneous=True)      # a homogeneous arm
    evaluate.build_test_datamodule(spec, max_datasets=30)

    assert captured["homogeneous_batches"] is False
    assert captured["batch"] == spec["batch_size"]


def test_missing_replicates_works_before_the_baseline_has_run():
    """The question matters most when nothing has run, so it must answer then.

    The signature comes from the baseline's *declared* spec rather than from a
    completed run; otherwise an empty pool could not be told what to run first.
    """
    assert stability.missing_replicates([], "F1a") == ["F1a", "F1b", "F1c"]


def test_missing_replicates_searches_the_frozen_family_too(tmp_path):
    """A Stage 0 pool re-read with --baseline E1a must still find E2b/E3b."""
    make_run(tmp_path, "E1a", spec_over={"seed": 101})
    rows, _ = collect.load_runs(tmp_path)
    assert stability.missing_replicates(rows, "E1a") == ["E2b", "E3b"]


def test_unknown_baseline_returns_nothing():
    assert stability.missing_replicates([], "NOPE") == []


# ---------------------------------------------------------------------------
# Confounded arms get no verdict
# ---------------------------------------------------------------------------
def test_spec_diff_names_the_knobs_that_differ():
    base = dict(BASE_SPEC)
    assert collect.spec_diff(base, dict(base)) == []
    assert collect.spec_diff(base, dict(base, lr=3e-3)) == ["lr"]
    assert collect.spec_diff(base, dict(base, seed=999)) == []   # seed is not a knob
    diffs = collect.spec_diff(base, dict(base, width=0.5, homogeneous=True))
    assert set(diffs) == {"width", "homogeneous"}


def test_spec_diff_ignores_derived_accumulate():
    """effective_batch and accumulate move together; counting both would make
    one conceptual change look like two and suppress its verdict."""
    base = dict(BASE_SPEC)
    other = dict(base, effective_batch=4096, accumulate=16)
    assert collect.spec_diff(base, other) == ["effective_batch"]


def test_multi_knob_arm_is_separated_and_gets_no_verdict(tmp_path):
    """A run differing in several knobs measures their sum, not any one."""
    make_run(tmp_path, "F1a", val_bps=0.52, spec_over={"seed": 201})
    make_run(tmp_path, "F1b", val_bps=0.53, spec_over={"seed": 202})
    make_run(tmp_path, "SINGLE", val_bps=0.60, spec_over={"lr": 3e-3})
    make_run(tmp_path, "MULTI", val_bps=0.70,
             spec_over={"lr": 3e-3, "width": 2.0, "homogeneous": True})

    rows, _ = collect.load_runs(tmp_path)
    result = stability.analyse(rows, "F1a")

    assert [a["run"] for a in result["arms"]] == ["SINGLE"]
    assert [a["run"] for a in result["confounded"]] == ["MULTI"]
    assert result["confounded"][0]["verdict_val_bps"] == "confounded"
    # These synthetic specs set model_config directly, so it does not track
    # width the way `launch.resolve` derives it; three knobs differ here.
    assert set(result["confounded"][0]["knobs"]) == {"lr", "width",
                                                     "homogeneous"}
