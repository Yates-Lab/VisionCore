#!/usr/bin/env python3
"""Stage source/model assets needed by the Figure 4 real-trace SSI scorer.

This bridges from an external VisionCore-style data tree into the clean
VisionCoreMain layout without importing code from that tree. By default it is a
dry run. Use --apply with --link-mode symlink for local validation, or
--link-mode copy for a self-contained local asset tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_"
    "medoidPosthocminRepcomplete0p45_movieMedoid"
)


@dataclass(frozen=True)
class SourceAsset:
    key: str
    source_rel: Path
    target_rel: Path
    expected_sha256: str | None
    note: str
    alias_rel: Path | None = None


ASSETS = (
    SourceAsset(
        key="direct_matrix_source_csv",
        source_rel=Path(
            "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
            "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
        ),
        target_rel=Path(
            "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
            "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
        ),
        expected_sha256="ac2364e22ede162940a9ba7de5c8ab2c2ef2ea9c9985d851e302769d17c12567",
        note="Reviewed BackImage image/FEM windows opened by the real-trace matrix scorer.",
    ),
    SourceAsset(
        key="window_features_csv",
        source_rel=Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review/window_features.csv"),
        target_rel=Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review/window_features.csv"),
        expected_sha256="e8e2fa28c39d4d0222502bbe73fc221210260212fbed25bdc6c2e6c6217f73ba",
        note="Raw fixation-window table used by Fig. 4 fixation-stat refresh stages and provenance checks.",
    ),
    SourceAsset(
        key="unit_tuning_csv",
        source_rel=Path(
            "outputs/active_sensing_movie_information/"
            "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
            "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
            "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
        ),
        target_rel=Path(
            "outputs/active_sensing_movie_information/"
            "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
            "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
            "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
        ),
        expected_sha256="7a506b617ccbda563cab1e7f10173f9015448f88ce71a1abec7b05dc8aaa92f2",
        note="RR100 SF/group metadata merged into unit_feature_table.csv.",
    ),
    SourceAsset(
        key="rr100_population_spec_json",
        source_rel=Path(
            "outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/"
            f"population_spec_{RR100_VERSION}.json"
        ),
        target_rel=Path(
            "outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/"
            f"population_spec_{RR100_VERSION}.json"
        ),
        expected_sha256="d599fb0718faa363520a91b8f0819edafbff74ec501899b590e4061fef557f08",
        note="RR100 population metadata for the movie-medoid reduced population.",
    ),
    SourceAsset(
        key="rr100_population_spec_npz",
        source_rel=Path(
            "outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/"
            f"population_spec_{RR100_VERSION}.npz"
        ),
        target_rel=Path(
            "outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/"
            f"population_spec_{RR100_VERSION}.npz"
        ),
        expected_sha256="ffdbf4deee0d2bf4cc82d1bb7363e4271ee61be2cb6e87962bf6909a5b80c3d4",
        note="RR100 population transform used after the full 756-channel twin readout.",
    ),
    SourceAsset(
        key="mcfarland_outputs_mono",
        source_rel=Path("outputs/artifacts/mcfarland/mcfarland_outputs_mono.pkl"),
        target_rel=Path("outputs/artifacts/mcfarland/mcfarland_outputs_mono.pkl"),
        expected_sha256=None,
        note="Large McFarland output metadata pickle used to assemble the canonical readout.",
        alias_rel=Path("scripts/mcfarland_outputs_mono.pkl"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="VisionCore-style tree containing the source assets.")
    parser.add_argument("--target-root", type=Path, default=ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/figures/fig4/provenance/real_trace_source_asset_stage_manifest.json",
    )
    parser.add_argument("--apply", action="store_true", help="Actually stage files. Omit for dry-run.")
    parser.add_argument("--force", action="store_true", help="Replace existing staged files/aliases.")
    parser.add_argument("--allow-missing", action="store_true", help="Return success even if source assets are absent.")
    parser.add_argument(
        "--verify-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check expected hashes for assets with known provenance.",
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
        help="Hash assets up to this size even when no expected hash is known.",
    )
    parser.add_argument("--link-mode", choices=("copy", "symlink", "hardlink"), default="copy")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated asset keys to stage. Defaults to all assets.",
    )
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


def should_hash(path: Path, expected_sha256: str | None, args: argparse.Namespace) -> bool:
    if expected_sha256 is not None and bool(args.verify_hashes):
        return True
    if bool(args.hash_large):
        return True
    try:
        return path.stat().st_size <= int(float(args.large_hash_threshold_mb) * 1024 * 1024)
    except FileNotFoundError:
        return False


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        raise IsADirectoryError(f"Refusing to replace directory: {path}")


def stage_file(source: Path, target: Path, *, link_mode: str, force: bool) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not bool(force):
            return "already-present"
        remove_existing(target)
    if link_mode == "copy":
        shutil.copy2(source, target)
        return "copied"
    if link_mode == "hardlink":
        os.link(source, target)
        return "hardlinked"
    if link_mode == "symlink":
        target.symlink_to(source)
        return "symlinked"
    raise ValueError(f"Unknown link mode: {link_mode}")


def create_alias(alias: Path, target: Path, *, force: bool, dry_run: bool) -> dict[str, Any]:
    rel_target = Path(os.path.relpath(target, alias.parent))
    exists = alias.exists() or alias.is_symlink()
    action = "would-create"
    if exists:
        try:
            current = alias.resolve(strict=False)
        except RuntimeError:
            current = None
        if current == target.resolve(strict=False):
            action = "already-present"
        elif not bool(force):
            return {
                "alias": alias,
                "target": target,
                "status": "blocked",
                "action": "exists-different-target",
            }
        else:
            action = "would-replace"
    if not dry_run and action != "already-present":
        alias.parent.mkdir(parents=True, exist_ok=True)
        if exists:
            remove_existing(alias)
        alias.symlink_to(rel_target)
        action = "created" if action == "would-create" else "replaced"
    return {"alias": alias, "target": target, "status": "ok", "action": action}


def selected_assets(args: argparse.Namespace) -> tuple[SourceAsset, ...]:
    if not str(args.only).strip():
        return ASSETS
    requested = {part.strip() for part in str(args.only).split(",") if part.strip()}
    known = {asset.key for asset in ASSETS}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown asset key(s): {', '.join(unknown)}. Known keys: {', '.join(sorted(known))}")
    return tuple(asset for asset in ASSETS if asset.key in requested)


def stage_asset(asset: SourceAsset, args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_root) / asset.source_rel
    target = Path(args.target_root) / asset.target_rel
    row: dict[str, Any] = {
        "key": asset.key,
        "source": source,
        "target": target,
        "note": asset.note,
        "expected_sha256": asset.expected_sha256,
        "exists": source.exists(),
        "size_bytes": source.stat().st_size if source.exists() and source.is_file() else None,
        "link_mode": str(args.link_mode),
        "dry_run": not bool(args.apply),
    }
    if not source.exists():
        row.update({"status": "missing-source", "action": "none"})
        return row

    observed_sha256 = None
    if should_hash(source, asset.expected_sha256, args):
        observed_sha256 = sha256_file(source)
    row["observed_sha256"] = observed_sha256
    if (
        bool(args.verify_hashes)
        and asset.expected_sha256 is not None
        and observed_sha256 is not None
        and observed_sha256 != asset.expected_sha256
    ):
        row.update({"status": "hash-mismatch", "action": "none"})
        return row

    if bool(args.apply):
        action = stage_file(source, target, link_mode=str(args.link_mode), force=bool(args.force))
    else:
        action = "would-stage" if not target.exists() else "already-present"
    row.update({"status": "ok", "action": action})
    if asset.alias_rel is not None:
        alias = Path(args.target_root) / asset.alias_rel
        row["alias"] = create_alias(alias, target, force=bool(args.force), dry_run=not bool(args.apply))
    return row


def main() -> int:
    args = parse_args()
    try:
        assets = selected_assets(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rows = [stage_asset(asset, args) for asset in assets]
    manifest = {
        "analysis": "fig4_real_trace_source_asset_stage",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": Path(args.source_root),
        "target_root": Path(args.target_root),
        "apply": bool(args.apply),
        "link_mode": str(args.link_mode),
        "assets": rows,
    }
    write_json(Path(args.manifest), manifest)

    print("FIGURE 4 REAL-TRACE SOURCE ASSET STAGE")
    print(f"source : {Path(args.source_root)}")
    print(f"target : {Path(args.target_root)}")
    print(f"mode   : {'apply' if bool(args.apply) else 'dry-run'} / {args.link_mode}")
    print(f"manifest: {Path(args.manifest)}")
    for row in rows:
        print(f"  [{row['status']}] {row['key']}: {row['action']}")
        if row.get("observed_sha256") is not None:
            print(f"      sha256: {row['observed_sha256']}")
        if row.get("alias"):
            alias = row["alias"]
            print(f"      alias: [{alias['status']}] {alias['alias']} -> {alias['target']} ({alias['action']})")

    bad_statuses = {"missing-source", "hash-mismatch"}
    if not bool(args.allow_missing) and any(row["status"] in bad_statuses for row in rows):
        return 2
    if any(isinstance(row.get("alias"), dict) and row["alias"].get("status") == "blocked" for row in rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
