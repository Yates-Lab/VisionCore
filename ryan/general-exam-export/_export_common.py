"""Shared helpers for the general-exam panel-data export.

These scripts exist so the oral-exam deck can render theme-matched foveal V1
figures on ``calcifer``, where neither the raw sessions (``/mnt/ssd/YatesMarmoV1``,
4.4 TB) nor the VisionCore environment are available. Each exporter imports the
*canonical* panel code from ``VisionCore/paper/figN/``, pulls out exactly the
arrays that panel plots, and writes them as a plain ``.npz`` — no pickle, no
VisionCore classes, so the receiving end needs only numpy.

Nothing here writes into ``paper/`` or into ``outputs/cache/``. The caches are
read-only inputs.

Every export is accompanied by a manifest entry recording the VisionCore commit
and the size/mtime of each cache file consumed, so a stale snapshot on the deck
side is detectable. (Size+mtime rather than a hash: the fig2 caches are 7-8 GB
each and hashing them costs more than the export itself.)
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EXPORT_ROOT = Path(__file__).resolve().parent
OUT_DIR = EXPORT_ROOT / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VISIONCORE_ROOT = EXPORT_ROOT.parents[1]          # .../VisionCore
PAPER_DIR = VISIONCORE_ROOT / "paper"


def add_paper_path(*figure_dirs: str) -> None:
    """Put ``VisionCore/paper`` and the named ``paper/<fig>`` dirs on sys.path.

    The panel modules import their siblings by bare name (``from _panel_common
    import ...``), so the specific figure directory has to be importable, not
    just the paper root.
    """
    for p in (str(PAPER_DIR), *(str(PAPER_DIR / d) for d in figure_dirs)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(VISIONCORE_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception as exc:                      # pragma: no cover - provenance only
        return f"unavailable: {exc}"


def _cache_stat(path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
    }


def save_panel(name: str, arrays: dict, *, source: str, caches=(), notes: str = "") -> Path:
    """Write ``arrays`` to ``out/<name>.npz`` and record a manifest entry.

    ``arrays`` values must be plain numpy-serializable objects; anything that
    would need ``allow_pickle`` on load is rejected here rather than at render
    time on the other machine.
    """
    clean = {}
    for key, value in arrays.items():
        arr = np.asarray(value)
        if arr.dtype == object:
            raise TypeError(
                f"{name}: array {key!r} has dtype=object and would need "
                "allow_pickle on load. Convert it to a numeric or unicode array."
            )
        clean[key] = arr

    out_path = OUT_DIR / f"{name}.npz"
    np.savez_compressed(out_path, **clean)

    entry = {
        "panel": name,
        "file": out_path.name,
        "bytes": out_path.stat().st_size,
        "source": source,
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                   for k, v in clean.items()},
        "caches": [_cache_stat(c) for c in caches],
        "notes": notes,
    }
    _append_manifest(entry)
    print(f"  wrote {out_path.name}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    return out_path


def _append_manifest(entry: dict) -> None:
    manifest_path = OUT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"panels": {}}
    manifest["visioncore_sha"] = _git_sha()
    manifest["exported_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["panels"][entry["panel"]] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
