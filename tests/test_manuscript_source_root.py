import hashlib
import json
import sys
from pathlib import Path


MANUSCRIPT = Path(__file__).resolve().parents[1] / "manuscript"
sys.path.insert(0, str(MANUSCRIPT))

import analysis_selection  # noqa: E402


def test_selected_analysis_reads_bundle_from_configured_source_root(tmp_path, monkeypatch):
    bundle = tmp_path / "outputs" / "selected"
    (bundle / "figure3/figures").mkdir(parents=True)
    (bundle / "figure4/production_figure4/figure").mkdir(parents=True)

    figure3 = bundle / "figure3/figures/figure3_manifest.json"
    figure4 = bundle / "figure4/production_figure4/figure/results_provenance.json"
    figure3.write_text("{}")
    figure4.write_text("{}")
    manifest = bundle / "FINAL_MANIFEST.json"
    manifest.write_text(json.dumps({"status": "complete", "checkpoint_sha256": "checkpoint"}))

    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "bundle": "outputs/selected",
        "checkpoint_sha256": "checkpoint",
        "final_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "figure3_manifest_sha256": hashlib.sha256(figure3.read_bytes()).hexdigest(),
        "figure4_results_sha256": hashlib.sha256(figure4.read_bytes()).hexdigest(),
    }))
    monkeypatch.setattr(analysis_selection, "SELECTION", selection)
    monkeypatch.setattr(analysis_selection, "SOURCE_ROOT", tmp_path)

    assert analysis_selection.selected_analysis()["bundle"] == "outputs/selected"


def test_source_path_relocates_recorded_repository_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis_selection, "SOURCE_ROOT", tmp_path)

    resolved = analysis_selection.source_path(
        "/home/jake/repos/VisionCore/outputs/selected/cache/result.pkl"
    )

    assert resolved == tmp_path / "outputs/selected/cache/result.pkl"
