from argparse import Namespace
from pathlib import Path

import pytest

from paper.fig4.spatiotemporal_tuning.run_production_figure4 import (
    build_commands,
    require_execution_environment,
    validate_panel_h_sample_size,
)
from paper.fig4.spatiotemporal_tuning.audit_revised_figure4_release import (
    expected_visual_audit_pages,
)


def test_execution_requires_yatesfv_only_when_execute_is_requested() -> None:
    require_execution_environment(execute=False, environment=None)
    require_execution_environment(execute=True, environment="yatesfv")
    with pytest.raises(RuntimeError, match="yatesfv"):
        require_execution_environment(execute=True, environment="base")


def test_release_refuses_smoke_sized_panel_h() -> None:
    validate_panel_h_sample_size(mode="smoke", n_movies=40)
    validate_panel_h_sample_size(mode="release", n_movies=100)
    with pytest.raises(ValueError, match="at least 100"):
        validate_panel_h_sample_size(mode="release", n_movies=40)


def test_visual_audit_pages_follow_the_fresh_validated_population() -> None:
    assert expected_visual_audit_pages({"n_validated_for_figure4": 145}) == list(
        range(1, 9)
    )
    assert expected_visual_audit_pages({"n_validated_for_figure4": 101}) == list(
        range(1, 7)
    )
    assert expected_visual_audit_pages(
        {
            "n_validated_for_figure4": 145,
            "visual_audit_contract": {
                "expected_validated_atlas_pages": [1, 2, 3]
            },
        }
    ) == [1, 2, 3]


def test_commands_preserve_explicit_all_unit_policy(tmp_path: Path) -> None:
    example_contract = tmp_path / "examples"
    example_contract.mkdir()
    (example_contract / "crossed_example_fits.csv").write_text(
        "unit_index,source_unit_index\n0,0\n", encoding="utf-8"
    )
    file_path = tmp_path / "input.file"
    file_path.write_text("x", encoding="utf-8")
    directory = tmp_path / "input-dir"
    directory.mkdir()
    paths = {
        "model_spec": file_path,
        "panel_a": directory,
        "panel_b": directory,
        "tuning_table": file_path,
        "tuning_summary": file_path,
        "all_fits": file_path,
        "example_contract": example_contract,
        "rucci": directory,
        "shards": [file_path],
        "trajectory": directory,
        "tuning_release": directory,
        "tuning_visual": file_path,
        "tuning_contract": directory,
        "population_spec": file_path,
        "passband": directory,
        "spectral_tuning": file_path,
    }
    args = Namespace(
        population_policy="all_checkpoint_available",
        n_bootstrap=100,
        seed=7,
    )
    render, audit, provenance = build_commands(
        args=args,
        paths=paths,
        figure_dir=tmp_path / "figure",
        audit_dir=tmp_path / "audit",
    )
    assert render[render.index("--population-policy") + 1] == "all_checkpoint_available"
    assert "--population-spec" in audit
    assert "--example-contract" in audit
    assert provenance[-1].endswith("results_provenance.json")
