"""Resolve the manuscript's explicit, completed analysis selection."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE_ROOT = Path(
    os.environ.get("VISIONCORE_MANUSCRIPT_SOURCE_ROOT", ROOT)
).expanduser().resolve()
SELECTION = HERE / 'analysis/selected_model_bundle.json'


def source_path(value):
    """Resolve recorded repository paths against the read-only source root."""
    path = Path(value)
    if not path.is_absolute():
        return SOURCE_ROOT / path
    if path.is_relative_to(SOURCE_ROOT):
        return path
    for anchor in ("outputs", "paper", "scripts"):
        if anchor in path.parts:
            return SOURCE_ROOT.joinpath(*path.parts[path.parts.index(anchor):])
    return path


def selected_analysis():
    if not SELECTION.exists():
        return {
            'bundle': 'outputs/clean_production_reproduction_f7b7e5e_20260827',
            'checkpoint_sha256': 'dd6780a8b34a5a280adb3926fa1662b229743d1a3b49affaf1590721ef153540',
            'schematic_no_phase_preview': True,
        }
    record = json.loads(SELECTION.read_text())
    manifest = SOURCE_ROOT / record['bundle'] / 'FINAL_MANIFEST.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != record['final_manifest_sha256']:
        raise ValueError('Selected analysis completion manifest changed')
    final = json.loads(manifest.read_text())
    if final['status'] != 'complete' or final['checkpoint_sha256'] != record['checkpoint_sha256']:
        raise ValueError('The manuscript selection requires a completed matching checkpoint')
    for key, relative in (
            ('figure3_manifest_sha256', 'figure3/figures/figure3_manifest.json'),
            ('figure4_results_sha256', 'figure4/production_figure4/figure/results_provenance.json')):
        if key in record:
            path = SOURCE_ROOT / record['bundle'] / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != record[key]:
                raise ValueError(f'Selected manuscript result changed: {path}')
    return record
