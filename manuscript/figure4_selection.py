"""Resolve the optional, hash-pinned C/F update while response replay is pending."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPECTRUM_SELECTION = HERE / 'analysis/figure4_spectrum_update.json'


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
