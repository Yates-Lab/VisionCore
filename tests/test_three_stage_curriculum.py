from pathlib import Path

import yaml

from training.run_three_stage_curriculum import (
    REPO_ROOT,
    build_stage_command,
    checkpoint_val_bps,
    select_handoff_checkpoint,
    validate_curriculum,
)


SPEC_PATH = REPO_ROOT / "experiments/curricula/native240_sparse_phase_v1.yaml"


def load_spec():
    return yaml.safe_load(SPEC_PATH.read_text())


def test_production_curriculum_contract():
    spec = load_spec()
    configs = validate_curriculum(spec)
    assert configs[0]["readout"]["type"] == "gaussian"
    assert configs[1]["trainable_components"] == ["readouts", "phase_readouts"]
    assert configs[2]["phase_readout"]["params"]["rank"] == 4


def test_commands_start_fresh_optimizers_at_boundaries(tmp_path):
    spec = load_spec()
    previous = tmp_path / "previous.ckpt"
    command = build_stage_command(spec, "TEST", 1, 0, tmp_path, previous)
    assert command[command.index("--pretrained_checkpoint") + 1] == str(previous)
    assert "--pretrained_load_heads" in command
    assert "--ckpt_path" not in command

    resume = tmp_path / "last.ckpt"
    command = build_stage_command(
        spec, "TEST", 1, 0, tmp_path, previous, resume_checkpoint=resume
    )
    assert command[command.index("--ckpt_path") + 1] == str(resume)
    assert "--pretrained_checkpoint" not in command


def test_checkpoint_handoff_policy(tmp_path):
    (tmp_path / "last.ckpt").touch()
    lower = tmp_path / "epoch=07-val_bps_overall=0.5800.ckpt"
    higher = tmp_path / "epoch=15-val_bps_overall=0.5900.ckpt"
    lower.touch()
    higher.touch()
    assert checkpoint_val_bps(higher) == 0.59
    assert select_handoff_checkpoint(tmp_path, "best_val_bps") == higher
    assert select_handoff_checkpoint(tmp_path, "last") == tmp_path / "last.ckpt"
