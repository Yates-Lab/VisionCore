from pathlib import Path

from paper.fig4.spatiotemporal_tuning.run_production_figure4 import (
    PRODUCTION_ENTRYPOINTS,
    production_source_closure,
)
from paper.model_selection.run_production_figure3 import (
    FIGURE3_ENTRYPOINTS,
    production_source_closure as figure3_source_closure,
)
from paper.production_source_closure import local_imports, source_closure
from training.run_three_stage_curriculum import (
    TRAINING_ENTRYPOINTS,
    production_source_closure as training_source_closure,
)


ROOT = Path(__file__).resolve().parents[1]


def test_local_import_resolver_handles_absolute_and_relative_imports(tmp_path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("from . import helper\n")
    (package / "helper.py").write_text("VALUE = 1\n")
    entry = package / "entry.py"
    entry.write_text("from pkg import helper\n")

    assert set(source_closure(tmp_path, (entry,))) == {
        (package / "__init__.py").resolve(),
        (package / "entry.py").resolve(),
        (package / "helper.py").resolve(),
    }
    assert (package / "helper.py").resolve() in local_imports(tmp_path, entry)


def test_figure4_source_closure_has_explicit_roots_and_no_directory_glob() -> None:
    closure = production_source_closure()
    assert set(PRODUCTION_ENTRYPOINTS) <= set(closure)
    source = (
        ROOT / "paper/fig4/spatiotemporal_tuning/run_production_figure4.py"
    ).read_text(encoding="utf-8")
    assert '.glob("*.py")' not in source


def test_orphaned_development_audit_is_not_a_figure4_dependency() -> None:
    relative = {str(path.relative_to(ROOT)) for path in production_source_closure()}
    assert (
        "paper/fig4/spatiotemporal_tuning/audit_output_unit_tuning_fits.py"
        not in relative
    )


def test_figure3_source_closure_includes_bare_sibling_imports() -> None:
    closure = figure3_source_closure()
    relative = {str(path.relative_to(ROOT)) for path in closure}
    assert set(FIGURE3_ENTRYPOINTS) <= set(closure)
    assert "paper/fig3/_fig3_data.py" in relative
    assert "paper/fig3/_fig3_ablation_data.py" in relative


def test_training_source_closure_reaches_actual_trainer() -> None:
    closure = training_source_closure()
    relative = {str(path.relative_to(ROOT)) for path in closure}
    assert set(TRAINING_ENTRYPOINTS) <= set(closure)
    assert "training/train_multidataset.py" in relative


def test_figure4_source_closure_uses_focused_primitives() -> None:
    relative = {str(path.relative_to(ROOT)) for path in production_source_closure()}
    expected = {
        "paper/fig4/spatiotemporal_tuning/eye_trace_filter.py",
        "paper/fig4/spatiotemporal_tuning/population_response.py",
        "paper/fig4/spatiotemporal_tuning/retinal_replay.py",
        "paper/fig4/spatiotemporal_tuning/spectral_power.py",
    }
    assert expected <= relative
