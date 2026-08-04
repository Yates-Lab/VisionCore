#!/usr/bin/env python3
"""Verify a Fig. 4 cache bundle in an isolated scratch cache.

The verifier extracts ``outputs/cache/*`` members from the bundle into a fresh
temporary cache, points VisionCore at that cache via environment variables,
composes Figure 4, and performs the same pixel-exact comparison as the
regression test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import fitz
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PDF = ROOT / "paper" / "fig4" / "reference" / "figure4_reference.pdf"
REFERENCE_MD5 = "48e092df402c7d43bc0e5e84fafb3d06"
ENTRY_POINT = ROOT / "paper" / "fig4" / "generate_figure4.py"
RENDER_DPI = 110


def _safe_cache_name(member_name: str) -> str | None:
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {member_name}")
    parts = path.parts
    if len(parts) != 3 or parts[0] != "outputs" or parts[1] != "cache":
        return None
    return parts[2]


def _extract_cache(bundle: Path, cache_dir: Path) -> dict[str, object]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    manifest: dict[str, object] | None = None
    with tarfile.open(bundle, "r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = _safe_cache_name(member.name)
            if name is None:
                continue
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read archive member: {member.name}")
            target = cache_dir / name
            with source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            extracted.append(name)
            if name == "fig4_cache_bundle_manifest.json":
                manifest = json.loads(target.read_text(encoding="utf-8"))
    return {"extracted": sorted(extracted), "manifest": manifest}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest_hashes(manifest: dict[str, object], cache_dir: Path) -> list[str]:
    errors: list[str] = []
    files = manifest.get("files", [])
    if not isinstance(files, list):
        return ["bundle manifest 'files' is not a list"]
    for record in files:
        if not isinstance(record, dict):
            errors.append("bundle manifest contains a non-object file record")
            continue
        name = record.get("name")
        expected = record.get("sha256")
        if not isinstance(name, str) or not isinstance(expected, str):
            errors.append(f"bundle manifest has an incomplete file record: {record!r}")
            continue
        path = cache_dir / name
        if not path.exists():
            errors.append(f"{name}: listed in manifest but not extracted")
            continue
        actual = _sha256(path)
        if actual != expected:
            errors.append(f"{name}: sha256 {actual} != {expected}")
    return errors


def _render(pdf_path: Path, dpi: int = RENDER_DPI) -> np.ndarray:
    with fitz.open(str(pdf_path)) as doc:
        pixmap = doc[0].get_pixmap(dpi=dpi)
    raster = np.frombuffer(pixmap.samples, dtype=np.uint8)
    raster = raster.reshape(pixmap.height, pixmap.width, pixmap.n)
    return raster[:, :, :3]


def _compare_pdfs(built_pdf: Path, work_dir: Path) -> dict[str, object]:
    reference_md5 = hashlib.md5(REFERENCE_PDF.read_bytes()).hexdigest()
    if reference_md5 != REFERENCE_MD5:
        raise RuntimeError(f"reference PDF md5 changed: {reference_md5} != {REFERENCE_MD5}")

    actual = _render(built_pdf)
    expected = _render(REFERENCE_PDF)
    result: dict[str, object] = {
        "reference_md5": reference_md5,
        "built_pdf": str(built_pdf),
        "render_dpi": RENDER_DPI,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
    }
    if actual.shape != expected.shape:
        result["max_channel_difference"] = None
        result["status"] = "shape-mismatch"
        return result
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    max_diff = int(diff.max())
    changed = int((diff.max(axis=2) > 0).sum())
    result["max_channel_difference"] = max_diff
    result["changed_pixels"] = changed
    result["status"] = "pass" if max_diff == 0 else "pixel-mismatch"
    if max_diff != 0:
        Image.fromarray(actual).save(work_dir / "figure4_built_110dpi.png")
        Image.fromarray(expected).save(work_dir / "figure4_reference_110dpi.png")
        heat = np.zeros_like(actual)
        heat[..., 0] = np.clip(diff.max(axis=2) * 8, 0, 255).astype(np.uint8)
        heat[..., 1] = actual[..., 1] // 3
        heat[..., 2] = expected[..., 2] // 3
        Image.fromarray(heat).save(work_dir / "figure4_diff_110dpi.png")
    return result


def _run_compose(work_dir: Path, cache_dir: Path, figures_dir: Path, stats_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "VISIONCORE_CACHE_DIR": str(cache_dir),
            "VISIONCORE_FIGURES_DIR": str(figures_dir),
            "VISIONCORE_STATS_DIR": str(stats_dir),
            "MPLCONFIGDIR": str(work_dir / "matplotlib"),
        }
    )
    return subprocess.run(
        [sys.executable, str(ENTRY_POINT)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--work-dir", type=Path, default=None, help="Scratch directory to use and keep.")
    parser.add_argument("--cleanup", action="store_true", help="Delete the scratch directory on success.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = args.bundle.expanduser().resolve()
    if not bundle.exists():
        raise SystemExit(f"bundle not found: {bundle}")

    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir is not None
        else Path(tempfile.mkdtemp(prefix="fig4_bundle_verify_"))
    )
    cache_dir = work_dir / "cache"
    figures_dir = work_dir / "figures"
    stats_dir = work_dir / "stats"
    work_dir.mkdir(parents=True, exist_ok=True)

    extraction = _extract_cache(bundle, cache_dir)
    manifest = extraction["manifest"]
    if not isinstance(manifest, dict):
        raise SystemExit("bundle does not contain outputs/cache/fig4_cache_bundle_manifest.json")
    required = [str(name) for name in manifest.get("required_inputs", [])]
    missing = [name for name in required if not (cache_dir / name).exists()]
    if missing:
        print(f"missing required inputs after extraction from {bundle}:")
        for name in missing:
            print(f"  {name}")
        return 2
    hash_errors = _validate_manifest_hashes(manifest, cache_dir)
    if hash_errors:
        print(f"bundle hash validation failed for {bundle}:")
        for error in hash_errors:
            print(f"  {error}")
        return 2

    compose = _run_compose(work_dir, cache_dir, figures_dir, stats_dir)
    built_pdf = figures_dir / "fig4" / "figure4.pdf"
    verification: dict[str, object] = {
        "bundle": str(bundle),
        "work_dir": str(work_dir),
        "cache_dir": str(cache_dir),
        "extracted_count": len(extraction["extracted"]),
        "required_inputs": len(required),
        "compose_returncode": compose.returncode,
        "compose_stdout": compose.stdout,
        "compose_stderr": compose.stderr,
    }
    if compose.returncode != 0:
        verification["status"] = "compose-failed"
        (work_dir / "fig4_cache_bundle_verify_manifest.json").write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(compose.stdout)
        print(compose.stderr, file=sys.stderr)
        print(f"verification work dir: {work_dir}")
        return compose.returncode
    if not built_pdf.exists():
        raise SystemExit(f"compose reported success but did not write {built_pdf}")

    comparison = _compare_pdfs(built_pdf, work_dir)
    verification.update(comparison)
    verify_manifest = work_dir / "fig4_cache_bundle_verify_manifest.json"
    verify_manifest.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"bundle          : {bundle}")
    print(f"work dir        : {work_dir}")
    print(f"required inputs : {len(required)}/{len(required)}")
    print(f"built pdf       : {built_pdf}")
    print(f"max diff        : {comparison['max_channel_difference']}")
    print(f"manifest        : {verify_manifest}")

    if comparison["status"] != "pass":
        print(f"verification failed: {comparison['status']}")
        return 1
    if args.cleanup:
        shutil.rmtree(work_dir)
        print("scratch cleanup : done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
