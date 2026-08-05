#!/usr/bin/env python3
"""Build a portable Fig. 4 cache bundle from the current flat cache.

The bundle is intentionally cache-first: it contains the canonical
``outputs/cache/fig4_*`` files needed to compose Figure 4 in a clean checkout,
plus a small set of refresh/provenance sidecars. It does not include the large
merged trace-bank directory or any raw data package.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths


@dataclass(frozen=True)
class BundleInput:
    path: Path
    role: str
    required: bool


SUPPORT_INPUTS = (
    BundleInput(_paths.PANEL_A_NETWORK_ICON_PROVENANCE_JSON, "support", False),
    BundleInput(_paths.BEHAVIOR_PATH_WINDOWS_CSV, "support", False),
    BundleInput(_paths.BRIDGE_MATCH_NULL_SUMMARY_CSV, "support", False),
    BundleInput(_paths.STORY_PANEL_B_SELECTION_SUMMARY_CSV, "support", False),
    BundleInput(_paths.STORY_PANEL_B_SUMMARY_JSON, "support", False),
)


def _relocate(path: Path, cache_dir: Path) -> Path:
    return cache_dir / path.name


def _bundle_inputs(cache_dir: Path, *, compose_only: bool) -> list[BundleInput]:
    entries: list[BundleInput] = [
        BundleInput(_relocate(path, cache_dir), "required", True)
        for path in _paths.REQUIRED_INPUTS
    ]
    if not compose_only:
        entries.extend(
            BundleInput(_relocate(path, cache_dir), "refresh-only", False)
            for path in _paths.REFRESH_ONLY_INPUTS
        )
        entries.extend(
            BundleInput(_relocate(entry.path, cache_dir), entry.role, entry.required)
            for entry in SUPPORT_INPUTS
        )

    by_name: dict[str, BundleInput] = {}
    for entry in entries:
        current = by_name.get(entry.path.name)
        if current is None or (entry.required and not current.required):
            by_name[entry.path.name] = entry
    return list(by_name.values())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(args: list[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return f"<external>/{resolved.name}"


def _archive_path(path: Path) -> str:
    return f"outputs/cache/{path.name}"


def _file_records(entries: Iterable[BundleInput]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    present: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for entry in entries:
        record = {
            "name": entry.path.name,
            "role": entry.role,
            "required": entry.required,
            "archive_path": _archive_path(entry.path),
            "refresh_source": _paths.REFRESH_SOURCES.get(_paths.CACHE_DIR / entry.path.name, "unknown"),
        }
        if entry.path.exists():
            record.update({
                "bytes": entry.path.stat().st_size,
                "sha256": _sha256(entry.path),
            })
            present.append(record)
        else:
            missing.append(record)
    return present, missing


def _manifest(
    *,
    bundle_name: str,
    cache_dir: Path,
    compose_only: bool,
    present: list[dict[str, object]],
    missing: list[dict[str, object]],
) -> dict[str, object]:
    status_short = _git_output(["git", "status", "--short"]).splitlines()
    return {
        "bundle": bundle_name,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "profile": "compose-only" if compose_only else "integration",
        "repo": {
            "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
            "git_status_short": status_short,
            "dirty": bool(status_short),
        },
        "cache_dir": _portable_path(cache_dir),
        "required_inputs": [path.name for path in _paths.REQUIRED_INPUTS],
        "refresh_only_inputs": [] if compose_only else [path.name for path in _paths.REFRESH_ONLY_INPUTS],
        "files": present,
        "missing": missing,
        "verify_command": f"uv run python paper/fig4/verify_cache_bundle.py {bundle_name}.tar.gz",
    }


def _add_bytes(tar: tarfile.TarFile, archive_path: str, data: bytes) -> None:
    info = tarfile.TarInfo(archive_path)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def _add_file(tar: tarfile.TarFile, path: Path, archive_path: str) -> None:
    info = tar.gettarinfo(str(path), archive_path)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with path.open("rb") as handle:
        tar.addfile(info, handle)


def _write_tarball(tar_path: Path, entries: list[BundleInput], manifest_bytes: bytes) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tar_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for entry in entries:
                    if entry.path.exists():
                        _add_file(tar, entry.path, _archive_path(entry.path))
                _add_bytes(tar, "outputs/cache/fig4_cache_bundle_manifest.json", manifest_bytes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=_paths.CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=_paths.FIG_DIR / "handoff")
    parser.add_argument("--name", default=None, help="Bundle stem; default includes today's UTC date.")
    parser.add_argument("--compose-only", action="store_true", help="Include only compose-required caches.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    date_stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    bundle_name = args.name or f"fig4_cache_bundle_{date_stamp}"
    out_dir = args.out_dir.expanduser().resolve()
    tar_path = out_dir / f"{bundle_name}.tar.gz"
    manifest_path = out_dir / f"{bundle_name}_manifest.json"
    checksum_path = out_dir / f"{bundle_name}_checksums.txt"

    entries = _bundle_inputs(cache_dir, compose_only=args.compose_only)
    present, missing = _file_records(entries)
    missing_required = [record for record in missing if record["required"]]
    if missing_required:
        print(f"missing required cache inputs in {cache_dir}:")
        for record in missing_required:
            print(f"  {record['name']}  <- {record['refresh_source']}")
        return 2

    manifest = _manifest(
        bundle_name=bundle_name,
        cache_dir=cache_dir,
        compose_only=args.compose_only,
        present=present,
        missing=missing,
    )
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_bytes)
    _write_tarball(tar_path, entries, manifest_bytes)

    checksum_lines = [
        f"{_sha256(tar_path)}  {tar_path.name}",
        f"{_sha256(manifest_path)}  {manifest_path.name}",
        "",
        "# Archive member checksums",
    ]
    for record in present:
        checksum_lines.append(f"{record['sha256']}  {record['archive_path']}")
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    print(f"bundle          : {tar_path}")
    print(f"manifest        : {manifest_path}")
    print(f"checksums       : {checksum_path}")
    print(f"included files  : {len(present)}")
    print(f"required inputs : {len(_paths.REQUIRED_INPUTS)}/{len(_paths.REQUIRED_INPUTS)}")
    if missing:
        print(f"missing optional: {len(missing)}")
        for record in missing:
            print(f"  {record['name']}")
    print(f"verify          : uv run python paper/fig4/verify_cache_bundle.py {tar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
