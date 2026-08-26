from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_figure3_runner_has_fail_closed_contract():
    source = (
        ROOT / "paper/model_selection/run_production_figure3.py"
    ).read_text()
    for key in (
        "production_model.yaml",
        "FIG3_MODEL_CHECKPOINT",
        "FIG3_DATASET_CONFIGS",
        "FIG3_REFERENCE_CACHE",
        "FIG3_COVDECOMP_CACHE_PATH",
        "FIG3_COVDECOMP_DERIVED_CACHE_PATH",
        "COVDECOMP_ALIGNED_CACHE_PATH",
        "FIG3_REUSE_EXISTING_CACHES",
        "FIG3_CACHE_PATH",
        "FIG3_ABLATION_CACHE_PATH",
        "FIG3_PANEL_A_CACHE_PATH",
        "FIG3_FIG_DIR",
        "FIG3_STAT_DIR",
        "regen_fig3_caches.py",
        "audit_ablation_cache.py",
        "generate_figure3.py",
        "analysis_code_sha256",
        "git_provenance",
        "canonical_observation_cache_sha256",
        "dataset_config_sha256",
        "checkpoint_sha256",
        "conda environment 'yatesfv'",
    ):
        assert key in source
    assert 'manifest["status"] = "failed"' in source
    assert "selected_model" not in source
    assert "M77" not in source


def test_figure3_core_uses_generic_model_environment_name():
    source = (ROOT / "paper/fig3/_fig3_data.py").read_text()
    assert "FIG3_MODEL_CHECKPOINT" in source
    assert "FIG3_TWIN_CHECKPOINT" not in source
    assert "fig3_model.pkl" in source
