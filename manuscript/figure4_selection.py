"""Resolve hash-pinned Figure 4 illustration and legacy spectral selections."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPECTRUM_SELECTION = HERE / 'analysis/figure4_spectrum_update.json'
EXAMPLE_SELECTION = HERE / 'analysis/figure4_example_selection.json'


def selected_example_dir(bundle):
    """Resolve the manuscript illustration independently of population results."""
    if not EXAMPLE_SELECTION.exists():
        return Path(bundle) / 'figure4/panel_a_exemplar_audit'
    selection = json.loads(EXAMPLE_SELECTION.read_text())
    if Path(bundle).resolve() != (ROOT / selection['model_bundle']).resolve():
        raise ValueError('Figure 4 example selection belongs to another model bundle')
    for name, expected in selection['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Figure 4 example source changed: {name}')
    directory = ROOT / selection['audit_dir']
    summary = json.loads((directory/'summary.json').read_text())
    if summary['checkpoint_sha256'] != selection['checkpoint_sha256']:
        raise ValueError('Figure 4 example checkpoint differs from its selection')
    return directory


def spectrum_update(bundle):
    if not SPECTRUM_SELECTION.exists():
        return None
    selection = json.loads(SPECTRUM_SELECTION.read_text())
    # A completed replacement bundle supersedes this interim selection.
    if Path(bundle).resolve() != (ROOT / selection['response_bundle']).resolve():
        return None
    for name, expected in selection['source_sha256'].items():
        path = ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Figure 4 spectrum source changed: {path}')
    if selection['updated_panels'] != ['C', 'F']:
        raise ValueError('The interim selection may replace only spectral panels C/F')
    return selection
