"""Check the current approved wording, data, graphics, fonts and rendered layout."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET

import fitz
import numpy as np
from PIL import Image

from fonts import HERE, checked_font_paths


def require(condition, message):
    if not condition:
        raise SystemExit(f'FAIL: {message}')


def main():
    checked_font_paths()
    project = tomllib.loads((HERE / 'pyproject.toml').read_text())['project']
    require(platform.python_version() == project['requires-python'].removeprefix('=='),
            'Use the pinned Python version via uv sync --frozen')
    versions = {}
    for dependency in project['dependencies']:
        name, expected = dependency.split('==')
        versions[name] = importlib.metadata.version(name)
        require(versions[name] == expected, f'{name}: expected {expected}, got {versions[name]}')

    # This snapshot is updated after reviewed changes, never by build.py.
    inputs = json.loads((HERE / 'reference/inputs.json').read_text())
    for name, expected in inputs.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                f'Non-local reference input: {name}')
        path = HERE / name
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f'Approved scientific/layout input changed: {name}')
        if path.suffix == '.svg':
            for element in ET.parse(path).iter():
                for key, value in element.attrib.items():
                    if key.rsplit('}', 1)[-1] == 'href':
                        require(value.startswith(('#', 'data:')), f'External SVG resource in {name}')

    with np.load(HERE / 'components/panel5/plot_data.npz', allow_pickle=False) as data:
        require(len(data['fano_unc']) == len(data['fano_cor']) == 19, 'Expected 19 session pairs')
        shapes = {k: list(data[k].shape) for k in data.files}
        require(all(np.isfinite(data[k]).all() for k in data.files), 'Nonfinite cached data')

    out = HERE / 'output'
    with fitz.open(HERE / 'reference/approved.pdf') as reference, \
            fitz.open(out / 'poster_typst_full_60x36.pdf') as current:
        require(len(current) == 1 and list(current[0].rect) == [0, 0, 4320, 2592], 'Wrong page size')
        require(reference[0].get_text() == current[0].get_text(), 'Extracted scientific wording differs')
        fonts = current[0].get_fonts(full=True)
        require(all(current.extract_font(f[0])[3] for f in fonts), 'A used font is not embedded')
        differences = {}
        # Compare strips at both preview and 144-dpi resolution without large
        # full-page arrays. Metadata and PDF object numbering do not affect this.
        for scale in (1, 2):
            changed = 0
            for y in range(0, 2592, 216):
                clip = fitz.Rect(0, y, 4320, min(y + 216, 2592))
                options = dict(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
                a = reference[0].get_pixmap(**options)
                b = current[0].get_pixmap(**options)
                changed += int(np.count_nonzero(np.any(
                    np.frombuffer(a.samples, dtype=np.uint8).reshape(-1, 3) !=
                    np.frombuffer(b.samples, dtype=np.uint8).reshape(-1, 3), axis=1)))
            differences[f'{72 * scale}_dpi'] = changed
        require(not any(differences.values()), f'Render differs from approved poster: {differences}')
        pix = current[0].get_pixmap(alpha=False)
        with Image.open(out / 'poster_typst_full.png') as png:
            require(png.size == (4320, 2592) and png.convert('RGB').tobytes() == pix.samples,
                    'Full PNG is missing, stale or differs from PDF')
        content = json.loads((HERE / 'poster_content.json').read_text())
        panels = [r for r in content['regions']
                  if r['name'] == 'intro' or r['name'].startswith('panel')]
        with fitz.open(out / 'poster_panels_actual_size.pdf') as deck:
            require(len(deck) == len(panels), 'Wrong panel-deck page count')
            for page, r in zip(deck, panels):
                require(abs(page.rect.width - r['width']) < .001 and
                        abs(page.rect.height - r['height']) < .001,
                        f"Panel deck resized {r['name']}")
                clip = fitz.Rect(r['x'], r['y'], r['x'] + r['width'], r['y'] + r['height'])
                original = current[0].get_text('words', clip=clip)
                exported = page.get_text('words')
                require([w[4] for w in original] == [w[4] for w in exported],
                        f"Panel-deck wording differs: {r['name']}")
                offset = np.array([r['x'], r['y'], r['x'], r['y']])
                require(np.allclose(np.array([w[:4] for w in original]) - offset,
                                    np.array([w[:4] for w in exported]), rtol=0, atol=.01),
                        f"Panel-deck text scaled or moved: {r['name']}")
        with Image.open(out / 'poster_typst_full.png') as png:
            for r in content['regions']:
                box = tuple(round(v) for v in (r['x'], r['y'], r['x'] + r['width'], r['y'] + r['height']))
                expected = png.crop(box)
                with Image.open(out / f"{r['name']}.png") as crop:
                    require(crop.size == expected.size and crop.tobytes() == expected.tobytes(),
                            f"Stale preview: {r['name']}")
    subprocess.run([sys.executable, str(HERE / 'audit_layout.py')], check=True)
    report = dict(page_size_pt=[4320, 2592], input_sha256_checks=len(inputs),
                  scientific_wording_exact=True, changed_pixels=differences,
                  panel_deck_pages=len(panels), panel_deck_original_scale=True,
                  embedded_fonts=sorted(set(f[3] for f in fonts)),
                  cached_data_shapes=shapes, python=platform.python_version(), dependencies=versions,
                  scope='Approved assets plus saved panel-5 results; no analysis or model fitting is rerun.')
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: approved data, wording, embedded fonts, PNGs and layout; zero changed pixels at 72 and 144 dpi')


if __name__ == '__main__':
    main()
