"""Verify the exact fonts used by both renderers; never silently substitute."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def font_manifest():
    return json.loads((HERE / "fonts/manifest.json").read_text())


def checked_font_paths():
    paths, problems = [], []
    for name, spec in font_manifest().items():
        folder = HERE / ("fonts/open" if spec["bundled"] else ".fonts")
        path = folder / name
        if not path.is_file():
            problems.append(f"Missing {name} ({spec['family']}, {spec['version']})")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            problems.append(f"Wrong font bytes: {name}; requires {spec['version']}")
        else:
            paths.append(path)
    if problems:
        raise SystemExit("Font dependency check failed:\n  " + "\n  ".join(problems)
                         + "\nRun: python setup_fonts.py --from-dir YOUR_LICENSED_FONT_DIRECTORY"
                         + "\nSee README.md. No fonts will be substituted or downloaded.")
    return paths
