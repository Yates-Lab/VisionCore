#!/usr/bin/env python3
"""Stage historical Fig. 4 handoff caches into the current flat cache layout.

The August 2026 handoff bundle preserved Declan's original output tree names.
This repo composes Fig. 4 from flat, canonical ``outputs/cache/fig4_*`` files.
This script is the explicit bridge between those two contracts: it copies only
known artifacts, records hashes, and reports which current required inputs are
still absent after staging.

Examples:
    uv run python paper/fig4/stage_cache_overlay.py \
        /path/to/ssi_figure_v4_cache_overlay.tar.gz

    uv run python paper/fig4/stage_cache_overlay.py /home/declan/VisionCore
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from VisionCore.paths import CACHE_DIR, VISIONCORE_ROOT

import _fig4_paths as _paths


@dataclass(frozen=True)
class CacheMapping:
    source: str
    target: Path
    note: str


CACHE_OVERLAY_MAPPINGS = (
    CacheMapping(
        "outputs/fig/ssi_figure_v2/panels/panel_bcef_path_bins_values.csv",
        _paths.STORY_PANEL_B_VALUES_CSV,
        "Displayed panels B/D path-bin values.",
    ),
    CacheMapping(
        "outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_values.csv",
        _paths.PATH_BINS_VALUES_CSV,
        "Displayed panel E option-sheet values.",
    ),
    CacheMapping(
        "outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_last_bin_contrasts.csv",
        _paths.PATH_BINS_LAST_BIN_CONTRASTS_CSV,
        "Displayed panel E final-bin contrasts.",
    ),
    CacheMapping(
        "outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_trace_bank_reference.csv",
        _paths.PATH_BINS_TRACE_BANK_REFERENCE_CSV,
        "Displayed panel E trace-bank reference.",
    ),
    CacheMapping(
        "outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_populations.csv",
        _paths.PATH_BINS_POPULATIONS_CSV,
        "Displayed panel E population definitions.",
    ),
    CacheMapping(
        "outputs/fig/ssi_figure_v2/panels/panel_h_unwrapped_edge_coherence_values.csv",
        _paths.EDGE_COHERENCE_PROFILES_CSV,
        "Displayed panel F edge-coherence profiles.",
    ),
    CacheMapping(
        "outputs/fig/ssi_figure_v2/panels/panel_h_unwrapped_edge_coherence_random_orientation_reference.csv",
        _paths.EDGE_COHERENCE_RANDOM_BASELINE_CSV,
        "Displayed panel F random-orientation baseline.",
    ),
    CacheMapping(
        "outputs/fig/ssi_figure_v2/panels_v3/panel_k_patch_radius_alignment_slope_values.csv",
        _paths.PATCH_RADIUS_ALIGNMENT_SLOPE_CSV,
        "Displayed panel H patch-radius slope values.",
    ),
    CacheMapping(
        "outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence_windows.csv",
        _paths.BEHAVIOR_PATH_WINDOWS_CSV,
        "Bridge refresh input.",
    ),
    CacheMapping(
        "outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_random_rotation_match_null_summary.csv",
        _paths.BRIDGE_MATCH_NULL_SUMMARY_CSV,
        "Bridge match-null summary.",
    ),
    CacheMapping(
        "outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_random_rotation_prediction_by_coherence_summary.csv",
        _paths.BRIDGE_PREDICTION_BY_COHERENCE_SUMMARY_CSV,
        "Displayed panel G coherence summary.",
    ),
    CacheMapping(
        "outputs/fig_ssi/trace_provenance/schematic_crop_real_backimage_trace_center40.csv",
        _paths.SCHEMATIC_TRACE_CENTER40_CSV,
        "Displayed panels A/C real trace excerpt.",
    ),
    CacheMapping(
        "outputs/fig_ssi/rr100_schematic_endpoint_final_maps/cache/schematic_rr100_final_maps.npz",
        _paths.SCHEMATIC_FINAL_MAPS_NPZ,
        "Displayed panels A/C endpoint maps.",
    ),
    CacheMapping(
        "outputs/fig_ssi/rr100_schematic_endpoint_final_maps/schematic_rr100_final_map_unit_metrics.csv",
        _paths.SCHEMATIC_FINAL_MAP_UNIT_METRICS_CSV,
        "Displayed panels A/C endpoint-map unit metrics.",
    ),
    CacheMapping(
        "outputs/fig/ssi_figure_v2/panels/cache/coherence_gallery.npz",
        _paths.COHERENCE_GALLERY_NPZ,
        "Displayed panel C coherence gallery.",
    ),
    CacheMapping(
        "outputs/fig/ssi_figure_v2/panels/cache/panel_a_layout_overrides.json",
        _paths.PANEL_A_LAYOUT_OVERRIDES_JSON,
        "Hand-tuned displayed panel A layout overrides.",
    ),
    CacheMapping(
        "outputs/fig/ssi_figure_v2/panels/cache/panel_d_layout_overrides.json",
        _paths.PANEL_D_LAYOUT_OVERRIDES_JSON,
        "Hand-tuned displayed panel C layout overrides.",
    ),
    CacheMapping(
        "outputs/fig/ssi_figure_v2/panels/cache/panel_a_network_icon.pdf",
        _paths.PANEL_A_NETWORK_ICON_PDF,
        "Displayed panel A network icon.",
    ),
)

LOWER_ROOT_MAPPINGS = (
    CacheMapping(
        "outputs/active_sensing_movie_information/backimage_rr100_instantaneous_unit_maps_latest_v1/cache/backimage_rr100_instantaneous_unit_maps.npz",
        _paths.UNIT_MAPS_NPZ,
        "RR100 instantaneous unit maps.",
    ),
    CacheMapping(
        "outputs/active_sensing_movie_information/backimage_rr100_instantaneous_unit_maps_latest_v1/cache/selected_patch.npy",
        _paths.UNIT_MAPS_SELECTED_PATCH_NPY,
        "RR100 selected image patch.",
    ),
    CacheMapping(
        "outputs/active_sensing_movie_information/backimage_rr100_instantaneous_unit_maps_latest_v1/displayed_movie_instantaneous_ssi_all_units.csv",
        _paths.UNIT_MAPS_SSI_ALL_UNITS_CSV,
        "RR100 per-unit SSI table.",
    ),
    CacheMapping(
        "outputs/active_sensing_movie_information/backimage_rr100_instantaneous_unit_maps_latest_v1/orientation_tuning_groups.csv",
        _paths.UNIT_MAPS_ORIENTATION_GROUPS_CSV,
        "RR100 orientation group assignments.",
    ),
    CacheMapping(
        "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/image_feature_table.csv",
        _paths.IMAGE_FEATURE_TABLE_CSV,
        "Merged real-trace image features.",
    ),
    CacheMapping(
        "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/trace_xy.npy",
        _paths.TRACE_XY_NPY,
        "Merged real-trace XY bank.",
    ),
    CacheMapping(
        "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/phase1_phase2_conditioning_v1/phase1_movie_analysis_table.csv",
        _paths.PHASE1_MOVIE_ANALYSIS_TABLE_CSV,
        "Geometry-story refresh input.",
    ),
    CacheMapping(
        "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/phase1_phase2_conditioning_v1/trace_component_conditioning_v1/phase2_contour_relative_trace_component_movie_metrics.csv",
        _paths.TRACE_COMPONENT_MOVIE_METRICS_CSV,
        "Contour-relative trace component movie metrics.",
    ),
    CacheMapping(
        "outputs/active_sensing_movie_information/backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/sf_group_ssi_modulation_dynamic_log_gaussian_marginal_tertile_v1/dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv",
        _paths.SF_TUNING_UNIT_GROUPS_CSV,
        "RR100 spatial-frequency tuning groups.",
    ),
    CacheMapping(
        "outputs/active_sensing_movie_information/backimage_trace_bank_diffusion_large_fixation_sample_n5000_n40_v1/filtered_path_length_le350arcmin/trace_bank_metadata_filtered.csv",
        _paths.TRACE_BANK_METADATA_FILTERED_CSV,
        "Geometry-story refresh input from the native trace-bank metadata generator.",
    ),
    CacheMapping(
        "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/phase1_phase2_conditioning_v1/plot_collections/backimage_real_trace_panel_b_cell_baseline_sf05_coh020_match15_values.csv",
        _paths.STORY_PANEL_B_VALUES_CSV,
        "Preferred CI-bearing producer cache for displayed panels B/D.",
    ),
    CacheMapping(
        "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/phase1_phase2_conditioning_v1/plot_collections/backimage_real_trace_panel_b_cell_baseline_sf05_coh020_match15_selection_summary.csv",
        _paths.STORY_PANEL_B_SELECTION_SUMMARY_CSV,
        "Selection summary for displayed panels B/D.",
    ),
    CacheMapping(
        "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/phase1_phase2_conditioning_v1/plot_collections/backimage_real_trace_panel_b_cell_baseline_sf05_coh020_match15_summary.json",
        _paths.STORY_PANEL_B_SUMMARY_JSON,
        "Summary metadata for displayed panels B/D.",
    ),
    CacheMapping(
        "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_position_spread_unwrapped_profiles_by_edge_coherence.csv",
        _paths.EDGE_COHERENCE_PROFILES_CSV,
        "Preferred producer-schema input for displayed panel F.",
    ),
    CacheMapping(
        "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_position_spread_random_orientation_baseline_by_edge_coherence.csv",
        _paths.EDGE_COHERENCE_RANDOM_BASELINE_CSV,
        "Preferred producer-schema random baseline for displayed panel F.",
    ),
)

MAPPINGS = CACHE_OVERLAY_MAPPINGS + LOWER_ROOT_MAPPINGS


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(VISIONCORE_ROOT))
    except ValueError:
        return str(path)


def _normalize_archive_name(name: str) -> str:
    posix = PurePosixPath(name)
    if posix.is_absolute():
        raise ValueError(f"unsafe archive member: {name}")
    parts = []
    for part in posix.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(f"unsafe archive member: {name}")
        parts.append(part)
    return "/".join(parts)


def _target_for(mapping: CacheMapping, cache_dir: Path) -> Path:
    return cache_dir / mapping.target.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_archive_member(tar: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    extracted = tar.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"could not read archive member: {member.name}")
    with extracted, target.open("wb") as out:
        shutil.copyfileobj(extracted, out)


def _stage_from_directory(
    source: Path,
    *,
    cache_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mapping in MAPPINGS:
        source_path = source / mapping.source
        target = _target_for(mapping, cache_dir)
        row: dict[str, object] = {
            "source": mapping.source,
            "target": _relative(target),
            "note": mapping.note,
        }
        if not source_path.is_file():
            row["status"] = "missing-source"
        elif target.exists() and not overwrite:
            row["status"] = "already-present"
            row["sha256"] = _sha256(target)
            row["bytes"] = target.stat().st_size
        elif dry_run:
            row["status"] = "would-copy"
            row["bytes"] = source_path.stat().st_size
        else:
            _copy_file(source_path, target)
            row["status"] = "copied"
            row["sha256"] = _sha256(target)
            row["bytes"] = target.stat().st_size
        rows.append(row)
    return rows


def _data_package_roots(source: Path, requested: list[Path]) -> list[Path]:
    roots = [path.expanduser().resolve() for path in requested]
    sibling = source.parent / "DataYatesV1"
    if sibling.exists():
        roots.append(sibling.resolve())
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root.exists() and root not in seen:
            seen.add(root)
            out.append(root)
    return out


def _can_import_data_yates(source: Path, requested_roots: list[Path]) -> bool:
    added: list[str] = []
    for root in _data_package_roots(source, requested_roots):
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
            added.append(root_str)
    try:
        __import__("DataYatesV1")
    except Exception:
        return False
    return True


def _stage_schematic_stimulus_payload(
    source: Path,
    *,
    cache_dir: Path,
    overwrite: bool,
    dry_run: bool,
    data_package_roots: list[Path],
) -> dict[str, object]:
    target = cache_dir / _paths.SCHEMATIC_STIMULUS_PAYLOAD_NPZ.name
    row: dict[str, object] = {
        "source": "derived from fig4_image_feature_table.csv plus BackImage data",
        "target": _relative(target),
        "note": "Displayed panels A/C cached stimulus canvas and crops.",
    }
    if target.exists() and not overwrite:
        row["status"] = "already-present"
        row["sha256"] = _sha256(target)
        row["bytes"] = target.stat().st_size
        return row

    image_table_ready = (
        (cache_dir / _paths.IMAGE_FEATURE_TABLE_CSV.name).exists()
        or (source / "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/image_feature_table.csv").exists()
    )
    trace_ready = (
        (cache_dir / _paths.SCHEMATIC_TRACE_CENTER40_CSV.name).exists()
        or (source / "outputs/fig_ssi/trace_provenance/schematic_crop_real_backimage_trace_center40.csv").exists()
    )
    can_build = source.is_dir() and image_table_ready and trace_ready and _can_import_data_yates(source, data_package_roots)
    if not can_build:
        row["status"] = "missing-source"
        return row
    if dry_run:
        row["status"] = "would-build"
        return row

    import _fig4_contour_schematic as schematic

    path = schematic.write_new_bank_stimulus_cache(target)
    row["status"] = "built"
    row["sha256"] = _sha256(path)
    row["bytes"] = path.stat().st_size
    return row


def _stage_from_tarball(
    source: Path,
    *,
    cache_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with tarfile.open(source, "r:*") as tar:
        members = {
            _normalize_archive_name(member.name): member
            for member in tar.getmembers()
            if member.isfile()
        }
        for mapping in MAPPINGS:
            target = _target_for(mapping, cache_dir)
            member = members.get(mapping.source)
            row: dict[str, object] = {
                "source": mapping.source,
                "target": _relative(target),
                "note": mapping.note,
            }
            if member is None:
                row["status"] = "missing-source"
            elif target.exists() and not overwrite:
                row["status"] = "already-present"
                row["sha256"] = _sha256(target)
                row["bytes"] = target.stat().st_size
            elif dry_run:
                row["status"] = "would-copy"
                row["bytes"] = member.size
            else:
                _copy_archive_member(tar, member, target)
                row["status"] = "copied"
                row["sha256"] = _sha256(target)
                row["bytes"] = target.stat().st_size
            rows.append(row)
    return rows


def _missing_required(
    cache_dir: Path,
    rows: list[dict[str, object]],
    *,
    assume_would_copy: bool,
) -> list[Path]:
    present_names = {
        Path(str(row["target"])).name
        for row in rows
        if row["status"] in {"already-present", "copied", "built"}
        or (assume_would_copy and row["status"] in {"would-copy", "would-build"})
    }
    missing = []
    for path in _paths.REQUIRED_INPUTS:
        target = cache_dir / path.name
        if path.name not in present_names and not target.exists():
            missing.append(target)
    return missing


def _write_manifest(
    manifest_path: Path,
    *,
    source: Path,
    cache_dir: Path,
    rows: list[dict[str, object]],
    missing_required: list[Path],
) -> None:
    payload = {
        "source": str(source),
        "cache_dir": str(cache_dir),
        "mapping_count": len(rows),
        "rows": rows,
        "required_inputs": len(_paths.REQUIRED_INPUTS),
        "missing_required_after_stage": [
            {
                "path": str(path),
                "name": path.name,
                "refresh_source": _paths.REFRESH_SOURCES.get(CACHE_DIR / path.name, "unknown"),
            }
            for path in missing_required
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage historical Fig. 4 handoff artifacts into outputs/cache/fig4_* names."
    )
    parser.add_argument("source", type=Path, help="Cache overlay tarball or a VisionCore-style source tree.")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="Target cache directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing staged cache files.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be copied without writing files.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any compose-required fig4 input is still missing after staging.",
    )
    parser.add_argument(
        "--manifest-name",
        default="fig4_cache_overlay_stage_manifest.json",
        help="Manifest filename written inside the target cache directory.",
    )
    parser.add_argument(
        "--data-package-root",
        type=Path,
        action="append",
        default=[],
        help="Optional checkout path to add to sys.path while deriving DataYates-backed caches.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"source not found: {source}")

    if source.is_dir():
        rows = _stage_from_directory(source, cache_dir=cache_dir, overwrite=args.overwrite, dry_run=args.dry_run)
        rows.append(
            _stage_schematic_stimulus_payload(
                source,
                cache_dir=cache_dir,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                data_package_roots=args.data_package_root,
            )
        )
    else:
        rows = _stage_from_tarball(source, cache_dir=cache_dir, overwrite=args.overwrite, dry_run=args.dry_run)

    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1

    missing_required = _missing_required(cache_dir, rows, assume_would_copy=args.dry_run)
    print(f"source          : {source}")
    print(f"target cache    : {cache_dir}")
    print("mapping status  : " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))
    print(f"required present: {len(_paths.REQUIRED_INPUTS) - len(missing_required)}/{len(_paths.REQUIRED_INPUTS)}")
    if missing_required:
        print("missing required:")
        for path in missing_required:
            print(f"  {path.name}  <- {_paths.REFRESH_SOURCES.get(CACHE_DIR / path.name, 'unknown')}")

    if not args.dry_run:
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / args.manifest_name
        _write_manifest(
            manifest_path,
            source=source,
            cache_dir=cache_dir,
            rows=rows,
            missing_required=missing_required,
        )
        print(f"manifest        : {manifest_path}")

    if args.strict and missing_required:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
