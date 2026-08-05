#!/usr/bin/env python3
"""Build a portable source-asset bundle for the Fig. 4 real-trace matrix.

The bundle is not a figure cache bundle. It contains the source/model-adjacent
artifacts the clean real-trace scorer needs in a VisionCoreMain checkout:
source window table, RR100 unit metadata, RR100 population spec, and the
McFarland readout artifact. The model checkpoint is included only when
--include-checkpoint is passed, and defaults to the staged local copy when it is
available.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any

UPSTREAM_DIR = Path(__file__).resolve().parent
ROOT = UPSTREAM_DIR.parents[2]

from stage_real_trace_source_assets import ASSETS, RR100_VERSION, SourceAsset  # noqa: E402


CHECKPOINT_ENV = "FIG4_TWIN_CHECKPOINT"
MODEL_CHECKPOINT_FILENAME = "epoch=147-val_bps_overall=0.5702.ckpt"
STAGED_CHECKPOINT_PATH = ROOT / "outputs/artifacts/model_checkpoints/fig4_twin" / MODEL_CHECKPOINT_FILENAME
CHECKPOINT_SHA256 = "55d084aa0beb7d65614aecb9122edf7ad49c5799d370dbbd5dcf60b815c62de3"
CHECKPOINT_ARCHIVE_PATH = Path("outputs/artifacts/model_checkpoints/fig4_twin/epoch=147-val_bps_overall=0.5702.ckpt")


def default_checkpoint_path() -> Path:
    if CHECKPOINT_ENV in os.environ:
        return Path(os.environ[CHECKPOINT_ENV])
    return STAGED_CHECKPOINT_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT,
        help="Tree containing the staged assets. Defaults to this VisionCoreMain checkout.",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/figures/fig4/handoff")
    parser.add_argument("--name", type=str, default=None, help="Bundle stem; default includes today's UTC date.")
    parser.add_argument("--only", type=str, default="", help="Comma-separated asset keys to include.")
    parser.add_argument("--allow-missing", action="store_true", help="Return success even if assets are absent.")
    parser.add_argument(
        "--verify-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify known asset hashes before packaging.",
    )
    parser.add_argument(
        "--hash-large",
        action="store_true",
        help="Hash large assets without expected hashes, including the McFarland pickle.",
    )
    parser.add_argument(
        "--large-hash-threshold-mb",
        type=float,
        default=256.0,
        help="Hash unpinned assets up to this size even when --hash-large is omitted.",
    )
    parser.add_argument(
        "--include-checkpoint",
        action="store_true",
        help="Also include the recovered model checkpoint in the archive.",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=default_checkpoint_path())
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_assets(only: str) -> tuple[SourceAsset, ...]:
    if not str(only).strip():
        return ASSETS
    requested = {part.strip() for part in str(only).split(",") if part.strip()}
    known = {asset.key for asset in ASSETS}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown asset key(s): {', '.join(unknown)}. Known keys: {', '.join(sorted(known))}")
    return tuple(asset for asset in ASSETS if asset.key in requested)


def should_hash(path: Path, expected_sha256: str | None, args: argparse.Namespace) -> bool:
    if expected_sha256 is not None and bool(args.verify_hashes):
        return True
    if bool(args.hash_large):
        return True
    try:
        return path.stat().st_size <= int(float(args.large_hash_threshold_mb) * 1024 * 1024)
    except FileNotFoundError:
        return False


def source_path_for_asset(source_root: Path, asset: SourceAsset) -> Path:
    candidates = (source_root / asset.target_rel, source_root / asset.source_rel)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def asset_record(asset: SourceAsset, source_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    source = source_path_for_asset(source_root, asset)
    row: dict[str, Any] = {
        "key": asset.key,
        "source": source,
        "archive_path": asset.target_rel,
        "expected_sha256": asset.expected_sha256,
        "note": asset.note,
        "exists": source.exists(),
        "include": False,
    }
    if not source.exists():
        row["status"] = "missing-source"
        return row
    observed_sha256 = sha256_file(source) if should_hash(source, asset.expected_sha256, args) else None
    row.update(
        {
            "size_bytes": source.stat().st_size,
            "observed_sha256": observed_sha256,
            "status": "ok",
            "include": True,
        }
    )
    if (
        bool(args.verify_hashes)
        and asset.expected_sha256 is not None
        and observed_sha256 is not None
        and observed_sha256 != asset.expected_sha256
    ):
        row["status"] = "hash-mismatch"
        row["include"] = False
    if asset.alias_rel is not None:
        row["alias"] = {
            "archive_path": asset.alias_rel,
            "target": asset.target_rel,
        }
    return row


def checkpoint_record(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.checkpoint_path)
    row: dict[str, Any] = {
        "key": "model_checkpoint",
        "source": path,
        "archive_path": CHECKPOINT_ARCHIVE_PATH,
        "expected_sha256": CHECKPOINT_SHA256,
        "note": "Optional recovered twin checkpoint. Large; not included unless --include-checkpoint is set.",
        "exists": path.exists(),
        "include": bool(args.include_checkpoint),
    }
    if not path.exists():
        row["status"] = "missing-source"
        row["include"] = False
        return row
    observed_sha256 = sha256_file(path) if bool(args.include_checkpoint) else None
    row.update({"size_bytes": path.stat().st_size, "observed_sha256": observed_sha256, "status": "ok"})
    if bool(args.include_checkpoint) and observed_sha256 != CHECKPOINT_SHA256:
        row["status"] = "hash-mismatch"
        row["include"] = False
    return row


def add_file(tar: tarfile.TarFile, path: Path, archive_path: Path) -> None:
    real_path = path.resolve(strict=True)
    info = tarfile.TarInfo(str(archive_path))
    info.size = real_path.stat().st_size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with real_path.open("rb") as handle:
        tar.addfile(info, handle)


def add_symlink(tar: tarfile.TarFile, archive_path: Path, target_path: Path) -> None:
    info = tarfile.TarInfo(str(archive_path))
    info.type = tarfile.SYMTYPE
    info.linkname = os.path.relpath(str(target_path), str(archive_path.parent))
    info.mode = 0o777
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tar.addfile(info)


def add_bytes(tar: tarfile.TarFile, archive_path: Path, data: bytes) -> None:
    info = tarfile.TarInfo(str(archive_path))
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def write_tarball(tar_path: Path, rows: list[dict[str, Any]], manifest_bytes: bytes) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tar_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for row in rows:
                    if not bool(row.get("include")):
                        continue
                    add_file(tar, Path(row["source"]), Path(row["archive_path"]))
                    alias = row.get("alias")
                    if isinstance(alias, dict):
                        add_symlink(tar, Path(alias["archive_path"]), Path(alias["target"]))
                add_bytes(tar, Path("outputs/figures/fig4/provenance/real_trace_source_asset_bundle_manifest.json"), manifest_bytes)


def checksum_display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    try:
        assets = selected_assets(str(args.only))
    except ValueError as exc:
        print(str(exc))
        return 2
    rows = [asset_record(asset, source_root, args) for asset in assets]
    if bool(args.include_checkpoint):
        rows.append(checkpoint_record(args))
    failures = [row for row in rows if row["status"] in {"missing-source", "hash-mismatch"}]
    if failures and not bool(args.allow_missing):
        print("Refusing to bundle because required source assets are not ready:")
        for row in failures:
            print(f"  [{row['status']}] {row['key']}: {row['source']}")
        return 2

    date_stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    bundle_name = args.name or f"fig4_real_trace_source_assets_{date_stamp}"
    out_dir = Path(args.out_dir).expanduser().resolve()
    tar_path = out_dir / f"{bundle_name}.tar.gz"
    manifest_path = out_dir / f"{bundle_name}_manifest.json"
    checksum_path = out_dir / f"{bundle_name}_checksums.txt"
    manifest = {
        "analysis": "fig4_real_trace_source_asset_bundle",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "bundle": tar_path.name,
        "source_root": source_root,
        "rr100_version": RR100_VERSION,
        "include_checkpoint": bool(args.include_checkpoint),
        "assets": rows,
        "install_command": f"tar -xzf {tar_path.name} -C /path/to/VisionCoreMain",
        "notes": [
            "Extract into a VisionCoreMain checkout to place assets at the paths expected by the in-repo launcher.",
            (
                "The model checkpoint is not included unless --include-checkpoint is passed; "
                "when included, the in-repo launcher auto-detects its staged path."
            ),
            "The archive includes scripts/mcfarland_outputs_mono.pkl as a symlink to the staged McFarland artifact.",
        ],
    }
    manifest_bytes = json.dumps(json_ready(manifest), indent=2, sort_keys=True).encode("utf-8") + b"\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_bytes)
    write_tarball(tar_path, rows, manifest_bytes)
    checksum_lines = [
        f"{sha256_file(tar_path)}  {checksum_display_path(tar_path, source_root)}",
        f"{sha256_file(manifest_path)}  {checksum_display_path(manifest_path, source_root)}",
    ]
    for row in rows:
        if bool(row.get("include")) and row.get("observed_sha256") is not None:
            checksum_lines.append(f"{row['observed_sha256']}  {row['archive_path']}")
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    included = [row for row in rows if bool(row.get("include"))]
    included_bytes = sum(int(row.get("size_bytes", 0)) for row in included)
    print(f"bundle        : {tar_path}")
    print(f"manifest      : {manifest_path}")
    print(f"checksums     : {checksum_path}")
    print(f"included files: {len(included)}")
    print(f"included bytes: {included_bytes}")
    if failures:
        print(f"missing/skipped failures: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
