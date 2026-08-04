"""Regenerate Figure 4's cached inputs in dependency order, into a scratch tree.

Why this exists
---------------
`_fig4_paths.REFRESH_SOURCES` names a producer for each cached input, but nothing
executes that mapping, and -- importantly -- *no refresh script writes a
`fig4_*` cache name at all*. Every reference to a `CACHE_DIR / "fig4_*"` path in
`refresh/` and `fixation_stats/` is a read. The producers write to
`outputs/figures/fig4/...` and `outputs/fig/ssi_figure_v2/panels/...` under their
own `OUT_STEM`, and someone copied the results into `outputs/cache/` by hand.

So this module supplies the two things the refresh path was missing:

1. the producer -> cache-name staging map (`STAGES[*].produces`), and
2. an execution order derived from the real inputs each producer reads.

It never writes to `CACHE_DIR`. Regenerated artefacts land under a scratch root
and are compared against a baseline copy of the caches with `--verify`. That
separation is the point: the 38 `fig4_*` files in `outputs/cache/` are untracked
and are the only copy of this figure's data.

Usage
-----
    uv run python paper/fig4/refresh_all.py                     # preflight only
    uv run python paper/fig4/refresh_all.py --run --scratch DIR
    uv run python paper/fig4/refresh_all.py --verify --scratch DIR --baseline DIR
    uv run python paper/fig4/refresh_all.py --manuscript-numbers --cache-dir DIR
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _fig4_paths as _paths  # noqa: E402
from VisionCore.paths import CACHE_DIR, VISIONCORE_ROOT as ROOT  # noqa: E402

FIG4_DIR = Path(__file__).resolve().parent
REFRESH_DIR = FIG4_DIR / "refresh"
FIXSTATS_DIR = FIG4_DIR / "fixation_stats"

# Upstream output trees the producers read. Neither is in this repo; both lived
# in a collaborator's home directory. Named here so the preflight can say which
# tree is missing rather than only which file.
UPSTREAM_FIXSTATS = ROOT / "outputs" / "fixation_statistics_by_stimulus_all_sessions_after_review"
UPSTREAM_MOVIE_INFO = ROOT / "outputs" / "active_sensing_movie_information"

# The merged real-trace SSI matrix. `_fig4_paths` declares it deliberately
# unstaged; every geometry-story producer reads it. These are the members
# `refresh/_fig4_trace_schematics.load_dataset` opens, so a staging attempt can
# be checked for completeness instead of failing partway through a run.
TRACE_BANK_MEMBERS = (
    "ssi_matrix.npy",
    "expected_spikes_matrix.npy",
    "stabilized_ssi_by_image.npy",
    "stabilized_expected_spikes_by_image.npy",
    "trace_xy.npy",
    "movie_feature_table.csv",
    "image_feature_table.csv",
    "trace_feature_table.csv",
    "unit_feature_table.csv",
    "stabilized_movie_feature_table.csv",
)

# Two caches `refresh/_fig4_geometry_story.py` reads that are absent from
# `outputs/cache/` and absent from REQUIRED_INPUTS -- an undeclared break the
# preflight would otherwise not mention.
UNDECLARED_INPUTS = (
    CACHE_DIR / "fig4_phase1_movie_analysis_table.csv",
    CACHE_DIR / "fig4_trace_bank_metadata_filtered.csv",
)


@dataclass
class Stage:
    """One producer, what it needs, and which cache names its outputs become."""

    key: str
    script: Path | None
    #: Named producer for stages whose script exists in no tree or git history.
    upstream_name: str | None = None
    #: Everything the producer opens. Missing entries block the stage.
    inputs: tuple[Path, ...] = ()
    #: Producer output filename (relative to its out-dir) -> cache path it is
    #: staged to. Empty for stages that only produce diagnostics.
    produces: dict[str, Path] = field(default_factory=dict)
    #: Where the producer writes when not redirected.
    default_out_dir: Path | None = None
    #: Whether the producer accepts `--out-dir`, i.e. can be pointed at scratch.
    out_dir_flag: str | None = None
    needs: tuple[str, ...] = ()
    note: str = ""

    def missing_inputs(self) -> list[Path]:
        return [p for p in self.inputs if not p.exists()]

    def status(self) -> tuple[str, str]:
        if self.script is None:
            return "NO-SCRIPT", f"producer '{self.upstream_name}' exists in no tree or git history"
        if not self.script.exists():
            return "NO-SCRIPT", f"{self.script.relative_to(FIG4_DIR)} not found"
        missing = self.missing_inputs()
        if missing:
            return "BLOCKED", f"{len(missing)} missing input(s): " + ", ".join(_short(p) for p in missing)
        if self.out_dir_flag is None:
            return "RUNNABLE-IN-PLACE", "no --out-dir; writes to its default out-dir"
        return "RUNNABLE", "all inputs present"


def _short(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


PANELS_V2 = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"
BRIDGE_OUT = _paths.FIG_DIR / "behavior_model_bridge"
COLLECTIONS_OUT = _paths.FIG_DIR / "plot_collections"

BACKIMAGE_WINDOWS_CSV = (
    UPSTREAM_FIXSTATS / "backimage_image_structure_reviewed_v2_screenfiltered_yfix" / "backimage_image_fem_windows.csv"
)
CONTOUR_MOTION_WINDOWS_CSV = (
    UPSTREAM_FIXSTATS / "backimage_contour_motion_component_plots_v1" / "contour_motion_component_windows.csv"
)
WINDOW_FEATURES_CSV = UPSTREAM_FIXSTATS / "window_features.csv"

_BANK = _paths.TRACE_BANK_MERGED_DIR
_BANK_INPUTS = tuple(_BANK / name for name in TRACE_BANK_MEMBERS)


STAGES: tuple[Stage, ...] = (
    # -- Tier 0: producers that exist nowhere ------------------------------
    Stage(
        key="rr100_spatial_ssi",
        script=None,
        upstream_name="run_backimage_contour_axis_rr100_spatial_ssi.py",
        produces={
            "unit_maps.npz": _paths.UNIT_MAPS_NPZ,
            "unit_maps_selected_patch.npy": _paths.UNIT_MAPS_SELECTED_PATCH_NPY,
            "unit_maps_ssi_all_units.csv": _paths.UNIT_MAPS_SSI_ALL_UNITS_CSV,
            "unit_maps_orientation_groups.csv": _paths.UNIT_MAPS_ORIENTATION_GROUPS_CSV,
        },
        note="RR100 movie run over the recordings; panel A/C instantaneous unit maps.",
    ),
    Stage(
        key="merge_ssi_shards",
        script=None,
        upstream_name="merge_backimage_real_trace_ssi_matrix_shards.py",
        produces={
            "image_feature_table.csv": _paths.IMAGE_FEATURE_TABLE_CSV,
            "trace_xy.npy": _paths.TRACE_XY_NPY,
        },
        note=(
            "Produces the merged trace bank at TRACE_BANK_MERGED_DIR; the two cache files "
            "are copies of members of that bank. Linchpin: every geometry-story stage needs it."
        ),
    ),
    Stage(
        key="phase1_phase2",
        script=None,
        upstream_name="analyze_backimage_real_trace_ssi_matrix_phase1_phase2.py",
        inputs=_BANK_INPUTS,
        produces={"trace_component_movie_metrics.csv": _paths.TRACE_COMPONENT_MOVIE_METRICS_CSV},
        needs=("merge_ssi_shards",),
    ),
    Stage(
        key="frequency_tuning_probe",
        script=None,
        upstream_name="run_backimage_rr100_frequency_tuning_probe.py",
        produces={"sf_tuning_unit_groups.csv": _paths.SF_TUNING_UNIT_GROUPS_CSV},
        note=(
            "Self-documenting output: frequency_tuning_contract and sf_split_metric_column "
            "record the formula, so the method is reconstructable from the cache if the script is not."
        ),
    ),
    Stage(
        key="schematic_final_maps",
        script=None,
        upstream_name="compute_schematic_rr100_final_maps.py",
        produces={
            "schematic_final_maps.npz": _paths.SCHEMATIC_FINAL_MAPS_NPZ,
            "schematic_final_map_unit_metrics.csv": _paths.SCHEMATIC_FINAL_MAP_UNIT_METRICS_CSV,
            "schematic_trace_center40.csv": _paths.SCHEMATIC_TRACE_CENTER40_CSV,
        },
        needs=("rr100_spatial_ssi",),
    ),
    Stage(
        key="matched_bins_bracket",
        script=None,
        upstream_name="make_backimage_panel_c_sf05_match15_matched_bins_bracket.py",
        inputs=_BANK_INPUTS,
        produces={
            "matched_bins_bracket_values.csv": _paths.MATCHED_BINS_BRACKET_VALUES_CSV,
            "matched_bins_bracket_last_bin_contrast.csv": _paths.MATCHED_BINS_BRACKET_LAST_BIN_CONTRAST_CSV,
            "matched_bins_bracket_summary.json": _paths.MATCHED_BINS_BRACKET_SUMMARY_JSON,
        },
        needs=("merge_ssi_shards",),
        note=(
            "refresh/_fig4_matched_bins_bracket.py is NOT this script: it reads these three "
            "caches to draw the bracket. It is a consumer, not a recovered producer."
        ),
    ),
    # -- Tier 1: in-repo producers reading the upstream fixation-stats tree --
    Stage(
        key="contour_motion_components",
        script=FIXSTATS_DIR / "plot_backimage_contour_motion_components.py",
        inputs=(WINDOW_FEATURES_CSV,),
        produces={
            "edge_coherence_profiles.csv": _paths.EDGE_COHERENCE_PROFILES_CSV,
            "edge_coherence_random_baseline.csv": _paths.EDGE_COHERENCE_RANDOM_BASELINE_CSV,
        },
        default_out_dir=UPSTREAM_FIXSTATS / "backimage_contour_motion_component_plots_v1",
        out_dir_flag="--out-dir",
        note="Panel F. Also emits contour_motion_component_windows.csv, which panel G's chain reads.",
    ),
    Stage(
        key="patch_radius_sensitivity",
        script=REFRESH_DIR / "summarize_backimage_patch_radius_sensitivity.py",
        inputs=(WINDOW_FEATURES_CSV,),
        produces={
            "patch_radius_alignment_slope_coherence_gt0p3.csv": _paths.PATCH_RADIUS_ALIGNMENT_SLOPE_CSV,
            "patch_radius_alignment_by_coherence.pdf": _paths.PATCH_RADIUS_ALIGNMENT_BY_COHERENCE_PDF,
        },
        default_out_dir=UPSTREAM_FIXSTATS,
        out_dir_flag="--root",
        note="Panel H.",
    ),
    Stage(
        key="coherence_gallery",
        script=REFRESH_DIR / "build_coherence_gallery_cache.py",
        inputs=(BACKIMAGE_WINDOWS_CSV,),
        produces={"coherence_gallery.npz": _paths.COHERENCE_GALLERY_NPZ},
        default_out_dir=PANELS_V2 / "cache",
        out_dir_flag=None,
        note="Panel C gallery. Hardcoded OUT_DIR; cannot be redirected without an edit.",
    ),
    Stage(
        key="behavior_path_windows",
        script=REFRESH_DIR / "behavior_component_path_by_coherence.py",
        inputs=(BACKIMAGE_WINDOWS_CSV, CONTOUR_MOTION_WINDOWS_CSV),
        produces={"behavior_component_path_by_coherence_windows.csv": _paths.BEHAVIOR_PATH_WINDOWS_CSV},
        default_out_dir=PANELS_V2,
        out_dir_flag="--out-dir",
        needs=("contour_motion_components",),
    ),
    # -- Tier 2: in-repo producers reading the merged trace bank ------------
    Stage(
        key="path_bins",
        script=REFRESH_DIR / "panel_g_alternative_x_axes_diagnostic.py",
        inputs=_BANK_INPUTS,
        produces={
            "panel_g_alternative_x_axes_diagnostic_values.csv": _paths.PATH_BINS_VALUES_CSV,
            "panel_g_alternative_x_axes_diagnostic_last_bin_contrasts.csv": _paths.PATH_BINS_LAST_BIN_CONTRASTS_CSV,
            "panel_g_alternative_x_axes_diagnostic_populations.csv": _paths.PATH_BINS_POPULATIONS_CSV,
            "panel_g_alternative_x_axes_diagnostic_trace_bank_reference.csv": _paths.PATH_BINS_TRACE_BANK_REFERENCE_CSV,
        },
        default_out_dir=PANELS_V2,
        out_dir_flag="--out-dir",
        needs=("merge_ssi_shards",),
        note="Panels B/D/E.",
    ),
    Stage(
        key="geometry_story_cde8bins",
        script=REFRESH_DIR / "_fig4_geometry_story_cde8bins.py",
        inputs=_BANK_INPUTS,
        produces={
            "backimage_real_trace_geometry_reordered_story_figure_cell_baseline_sf075_coh020_cde8bins_component_values.csv": _paths.STORY_COMPONENT_VALUES_CSV,
            "backimage_real_trace_geometry_reordered_story_figure_cell_baseline_sf075_coh020_cde8bins_selection_summary.csv": _paths.STORY_COMPONENT_SELECTION_SUMMARY_CSV,
            "backimage_real_trace_geometry_reordered_story_figure_cell_baseline_sf075_coh020_cde8bins_summary.json": _paths.STORY_COMPONENT_SUMMARY_JSON,
        },
        default_out_dir=COLLECTIONS_OUT,
        out_dir_flag=None,
        needs=("merge_ssi_shards",),
    ),
    Stage(
        key="orientation_match_sf05",
        script=REFRESH_DIR / "_fig4_orientation_match_sf05.py",
        inputs=_BANK_INPUTS,
        produces={
            "backimage_real_trace_panel_b_cell_baseline_sf05_coh020_match15_values.csv": _paths.STORY_PANEL_B_VALUES_CSV,
            "backimage_real_trace_panel_b_cell_baseline_sf05_coh020_match15_selection_summary.csv": _paths.STORY_PANEL_B_SELECTION_SUMMARY_CSV,
            "backimage_real_trace_panel_b_cell_baseline_sf05_coh020_match15_summary.json": _paths.STORY_PANEL_B_SUMMARY_JSON,
        },
        default_out_dir=COLLECTIONS_OUT,
        out_dir_flag=None,
        needs=("merge_ssi_shards", "geometry_story_cde8bins"),
        note="Thin wrapper that overrides OUT_STEM on _fig4_orientation_match.",
    ),
    # -- Tier 3: producers reading only staged caches -----------------------
    Stage(
        key="bridge_by_coherence",
        script=REFRESH_DIR / "_fig4_bridge_by_coherence.py",
        inputs=(
            _paths.BEHAVIOR_PATH_WINDOWS_CSV,
            _paths.PATH_BINS_VALUES_CSV,
            _paths.PATH_BINS_POPULATIONS_CSV,
        ),
        produces={
            "behavior_model_bridge_random_rotation_prediction_by_coherence_summary.csv": _paths.BRIDGE_PREDICTION_BY_COHERENCE_SUMMARY_CSV
        },
        default_out_dir=BRIDGE_OUT,
        out_dir_flag="--out-dir",
        needs=("behavior_path_windows", "path_bins"),
        note="Panel G.",
    ),
    Stage(
        key="bridge_rotation_null",
        script=REFRESH_DIR / "_fig4_bridge_rotation_null.py",
        inputs=(
            _paths.BEHAVIOR_PATH_WINDOWS_CSV,
            _paths.PATH_BINS_VALUES_CSV,
            _paths.PATH_BINS_POPULATIONS_CSV,
        ),
        produces={
            "behavior_model_bridge_random_rotation_match_null_summary.csv": _paths.BRIDGE_MATCH_NULL_SUMMARY_CSV
        },
        default_out_dir=BRIDGE_OUT,
        out_dir_flag="--out-dir",
        needs=("behavior_path_windows", "path_bins"),
    ),
    # -- Tier 4: self-contained ---------------------------------------------
    Stage(
        key="network_icon",
        script=FIG4_DIR / "_fig4_network_icon.py",
        inputs=(_paths.LAYOUT_REFERENCE_PDF,),
        produces={
            "fig4_panel_a_network_icon.pdf": _paths.PANEL_A_NETWORK_ICON_PDF,
            "fig4_panel_a_network_icon_provenance.json": _paths.PANEL_A_NETWORK_ICON_PROVENANCE_JSON,
        },
        default_out_dir=CACHE_DIR,
        out_dir_flag=None,
        note="Extracts the icon from the Illustrator reference. Writes into CACHE_DIR directly.",
    ),
)

# Hand-tuned, declared not regenerated. Listed so the report accounts for all 23
# required inputs rather than silently covering 21.
NOT_REGENERATED = {
    _paths.PANEL_A_LAYOUT_OVERRIDES_JSON: "hand-tuned layout overrides",
    _paths.PANEL_D_LAYOUT_OVERRIDES_JSON: "hand-tuned layout overrides",
}


def _order() -> list[Stage]:
    by_key = {s.key: s for s in STAGES}
    seen: set[str] = set()
    out: list[Stage] = []

    def visit(stage: Stage, trail: tuple[str, ...] = ()) -> None:
        if stage.key in seen:
            return
        if stage.key in trail:
            raise RuntimeError(f"dependency cycle: {' -> '.join(trail + (stage.key,))}")
        for dep in stage.needs:
            visit(by_key[dep], trail + (stage.key,))
        seen.add(stage.key)
        out.append(stage)

    for stage in STAGES:
        visit(stage)
    return out


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight() -> int:
    print("=" * 100)
    print("FIGURE 4 REFRESH PREFLIGHT")
    print("=" * 100)

    print(f"\nrepo root      : {ROOT}")
    print(f"cache dir      : {CACHE_DIR}")
    print(f"required inputs: {len(_paths.REQUIRED_INPUTS)}  missing: {len(_paths.missing_inputs())}")

    print("\n-- upstream trees the producers read " + "-" * 62)
    for tree in (UPSTREAM_FIXSTATS, UPSTREAM_MOVIE_INFO):
        print(f"  [{'present' if tree.exists() else 'ABSENT ':7}] {_short(tree)}")

    print("\n-- merged trace bank " + "-" * 78)
    if not _BANK.exists():
        print(f"  [ABSENT ] {_short(_BANK)}  ({len(TRACE_BANK_MEMBERS)} members expected)")
    else:
        have = [m for m in TRACE_BANK_MEMBERS if (_BANK / m).exists()]
        print(f"  [{'present' if len(have) == len(TRACE_BANK_MEMBERS) else 'PARTIAL'}] "
              f"{_short(_BANK)}  {len(have)}/{len(TRACE_BANK_MEMBERS)} members")
        for member in TRACE_BANK_MEMBERS:
            if not (_BANK / member).exists():
                print(f"            missing member: {member}")

    print("\n-- caches read by producers but absent and undeclared " + "-" * 45)
    for path in UNDECLARED_INPUTS:
        print(f"  [{'present' if path.exists() else 'ABSENT ':7}] {path.name}   <- refresh/_fig4_geometry_story.py")

    print("\n-- stages, in dependency order " + "-" * 68)
    counts: dict[str, int] = {}
    for stage in _order():
        status, detail = stage.status()
        counts[status] = counts.get(status, 0) + 1
        name = stage.script.name if stage.script else f"{stage.upstream_name}  (upstream)"
        print(f"\n  [{status:17}] {stage.key}")
        print(f"      producer : {name}")
        print(f"      reason   : {detail}")
        if stage.produces:
            for out_name, cache_path in stage.produces.items():
                mark = "present" if cache_path.exists() else "ABSENT"
                print(f"      stages   : {out_name}")
                print(f"                 -> {cache_path.name}  [{mark} in cache]")
        if stage.note:
            print(f"      note     : {stage.note}")

    print("\n" + "-" * 100)
    print("STAGE SUMMARY: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    covered = {p for s in STAGES for p in s.produces.values()}
    unaccounted = [p for p in _paths.REQUIRED_INPUTS if p not in covered and p not in NOT_REGENERATED]
    print(f"required inputs covered by a stage : {len([p for p in _paths.REQUIRED_INPUTS if p in covered])}"
          f"/{len(_paths.REQUIRED_INPUTS)}")
    print(f"required inputs declared not regenerated : {len(NOT_REGENERATED)}")
    if unaccounted:
        print("UNACCOUNTED required inputs (no stage, not declared hand-tuned):")
        for path in unaccounted:
            print(f"  {path.name}")

    regenerable = [p for s in _order() if s.status()[0] in ("RUNNABLE", "RUNNABLE-IN-PLACE")
                   for p in s.produces.values()]
    print(f"\nrequired inputs reachable right now : "
          f"{len([p for p in _paths.REQUIRED_INPUTS if p in set(regenerable)])}/{len(_paths.REQUIRED_INPUTS)}")
    print("=" * 100)
    return 0


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(scratch: Path, only: set[str] | None, allow_in_place: bool) -> int:
    scratch = scratch.resolve()
    if _is_under(scratch, CACHE_DIR):
        raise SystemExit(f"refusing to run: scratch {scratch} is inside the cache dir {CACHE_DIR}")
    scratch.mkdir(parents=True, exist_ok=True)
    staged_dir = scratch / "staged_cache"
    staged_dir.mkdir(exist_ok=True)

    results: list[dict] = []
    for stage in _order():
        if only and stage.key not in only:
            continue
        status, detail = stage.status()
        if status in ("NO-SCRIPT", "BLOCKED"):
            results.append({"stage": stage.key, "result": status, "detail": detail})
            print(f"[skip {status:9}] {stage.key}: {detail}")
            continue
        if status == "RUNNABLE-IN-PLACE" and not allow_in_place:
            results.append({
                "stage": stage.key,
                "result": "SKIPPED-IN-PLACE",
                "detail": f"no --out-dir flag; would write to {_short(stage.default_out_dir)}. "
                          f"Pass --allow-in-place to run anyway.",
            })
            print(f"[skip in-place  ] {stage.key}: no --out-dir flag")
            continue

        out_dir = scratch / stage.key
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(stage.script)]
        if stage.out_dir_flag:
            cmd += [stage.out_dir_flag, str(out_dir)]
            effective_out = out_dir
        else:
            effective_out = stage.default_out_dir
            if _is_under(effective_out, CACHE_DIR):
                results.append({
                    "stage": stage.key,
                    "result": "REFUSED",
                    "detail": f"default out-dir {_short(effective_out)} is inside CACHE_DIR; "
                              f"running it would overwrite baseline caches",
                })
                print(f"[REFUSED        ] {stage.key}: default out-dir is inside CACHE_DIR")
                continue

        print(f"[run            ] {stage.key}: {' '.join(cmd)}")
        env = dict(os.environ, MPLBACKEND="Agg")
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
            results.append({"stage": stage.key, "result": "FAILED",
                            "detail": f"exit {proc.returncode}", "stderr": "\n".join(tail)})
            print(f"[FAILED         ] {stage.key}: exit {proc.returncode}")
            for line in tail:
                print(f"    | {line}")
            continue

        staged, absent = [], []
        for out_name, cache_path in stage.produces.items():
            src = effective_out / out_name
            if src.exists():
                shutil.copy2(src, staged_dir / cache_path.name)
                staged.append(cache_path.name)
            else:
                absent.append(out_name)
        results.append({"stage": stage.key, "result": "OK", "staged": staged, "not_produced": absent})
        print(f"[OK             ] {stage.key}: staged {len(staged)}, expected-but-absent {len(absent)}")
        for name in absent:
            print(f"    ! producer did not write {name}")

    report = scratch / "refresh_all_report.json"
    report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 100)
    print("RUN SUMMARY")
    print("=" * 100)
    by_result: dict[str, list[str]] = {}
    for row in results:
        by_result.setdefault(row["result"], []).append(row["stage"])
    for result, keys in sorted(by_result.items()):
        print(f"  {result:17} {len(keys):2}  {', '.join(keys)}")
    staged_names = {n for row in results for n in row.get("staged", [])}
    print(f"\nstaged into {staged_dir}: {len(staged_names)} file(s)")
    print(f"required inputs regenerated: "
          f"{len([p for p in _paths.REQUIRED_INPUTS if p.name in staged_names])}/{len(_paths.REQUIRED_INPUTS)}")
    print(f"report: {report}")
    return 0


def _is_under(path: Path | None, parent: Path) -> bool:
    if path is None:
        return False
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify(scratch: Path, baseline: Path, rtol: float, atol: float) -> int:
    """Per-file comparison of regenerated caches against the baseline copy.

    Reports one line per file. A file the run never produced is reported as
    NOT-REGENERATED, not quietly omitted -- the whole point of this step is that
    a partial refresh must not read as a passing one.
    """
    import numpy as np
    import pandas as pd

    staged_dir = scratch / "staged_cache"
    rows: list[tuple[str, str, str]] = []

    for cache_path in _paths.REQUIRED_INPUTS:
        name = cache_path.name
        new = staged_dir / name
        old = baseline / name
        if not old.exists():
            rows.append((name, "NO-BASELINE", f"{old} absent"))
            continue
        if not new.exists():
            rows.append((name, "NOT-REGENERATED", "no stage produced this file in the run"))
            continue
        rows.append((name, *_compare(new, old, rtol, atol, np, pd)))

    width = max(len(r[0]) for r in rows)
    print("=" * 100)
    print(f"VERIFY  scratch={staged_dir}  baseline={baseline}  rtol={rtol} atol={atol}")
    print("=" * 100)
    for name, verdict, detail in rows:
        print(f"  {verdict:16} {name:{width}}  {detail}")

    counts: dict[str, int] = {}
    for _, verdict, _ in rows:
        counts[verdict] = counts.get(verdict, 0) + 1
    print("\n" + "-" * 100)
    print("VERIFY SUMMARY: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"files compared: {counts.get('PASS', 0) + counts.get('FAIL', 0)}/{len(_paths.REQUIRED_INPUTS)}")
    print("=" * 100)
    return 1 if counts.get("FAIL") or counts.get("NOT-REGENERATED") else 0


def _compare(new: Path, old: Path, rtol: float, atol: float, np, pd) -> tuple[str, str]:
    suffix = new.suffix.lower()
    try:
        if suffix == ".csv":
            a, b = pd.read_csv(new), pd.read_csv(old)
            if list(a.columns) != list(b.columns):
                return "FAIL", f"columns differ: {set(a.columns) ^ set(b.columns)}"
            if a.shape != b.shape:
                return "FAIL", f"shape {a.shape} vs {b.shape}"
            worst_col, worst = None, 0.0
            for col in a.columns:
                if pd.api.types.is_numeric_dtype(b[col]):
                    x, y = a[col].to_numpy(float), b[col].to_numpy(float)
                    finite = np.isfinite(x) & np.isfinite(y)
                    if (np.isfinite(x) != np.isfinite(y)).any():
                        return "FAIL", f"{col}: NaN/inf pattern differs"
                    if finite.any():
                        dev = np.abs(x[finite] - y[finite]) / np.maximum(np.abs(y[finite]), atol)
                        if dev.max() > worst:
                            worst, worst_col = float(dev.max()), col
                    if not np.allclose(x[finite], y[finite], rtol=rtol, atol=atol):
                        return "FAIL", f"{col}: max rel dev {float(np.abs(x[finite]-y[finite]).max()):.3g}"
                else:
                    if not a[col].astype(str).equals(b[col].astype(str)):
                        return "FAIL", f"{col}: non-numeric values differ"
            return "PASS", f"{a.shape[0]}x{a.shape[1]}, worst rel dev {worst:.2e}" + (f" ({worst_col})" if worst_col else "")

        if suffix == ".npy":
            a, b = np.load(new, allow_pickle=False), np.load(old, allow_pickle=False)
            if a.shape != b.shape:
                return "FAIL", f"shape {a.shape} vs {b.shape}"
            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                return "FAIL", f"max abs dev {float(np.nanmax(np.abs(a - b))):.3g}"
            return "PASS", f"shape {a.shape}"

        if suffix == ".npz":
            a, b = np.load(new, allow_pickle=False), np.load(old, allow_pickle=False)
            if set(a.files) != set(b.files):
                return "FAIL", f"keys differ: {set(a.files) ^ set(b.files)}"
            for key in b.files:
                x, y = a[key], b[key]
                if x.shape != y.shape:
                    return "FAIL", f"{key}: shape {x.shape} vs {y.shape}"
                if x.dtype.kind in "fc":
                    if not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
                        return "FAIL", f"{key}: max abs dev {float(np.nanmax(np.abs(x - y))):.3g}"
                elif not np.array_equal(x, y):
                    return "FAIL", f"{key}: values differ"
            return "PASS", f"{len(b.files)} array(s)"

        if suffix == ".json":
            bad = _json_diff(json.loads(new.read_text()), json.loads(old.read_text()), rtol, atol, "")
            return ("FAIL", bad) if bad else ("PASS", "")

        if suffix == ".pdf":
            # Vector output is not byte-reproducible (timestamps, font subset ids).
            return "SKIP-BINARY", f"pdf, {new.stat().st_size} vs {old.stat().st_size} bytes"

        return "SKIP-BINARY", f"no comparator for {suffix}"
    except Exception as exc:  # noqa: BLE001 - the comparison itself failing is a result
        return "FAIL", f"comparison raised {type(exc).__name__}: {exc}"


def _json_diff(a, b, rtol: float, atol: float, path: str) -> str:
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(a) != set(b):
            return f"{path or '<root>'}: keys differ"
        for key in b:
            bad = _json_diff(a[key], b[key], rtol, atol, f"{path}.{key}")
            if bad:
                return bad
        return ""
    if isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            return f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            bad = _json_diff(x, y, rtol, atol, f"{path}[{i}]")
            if bad:
                return bad
        return ""
    if isinstance(b, bool) or b is None:
        return "" if a == b else f"{path}: {a!r} vs {b!r}"
    if isinstance(b, (int, float)):
        if not isinstance(a, (int, float)):
            return f"{path}: type differs"
        if abs(a - b) > atol + rtol * abs(b):
            return f"{path}: {a!r} vs {b!r}"
        return ""
    return "" if a == b else f"{path}: {a!r} vs {b!r}"


# ---------------------------------------------------------------------------
# Manuscript numbers
# ---------------------------------------------------------------------------

#: Numbers quoted in the manuscript, and where each is read from. Locations
#: marked `None` were not resolved to a single cache field; they are reported as
#: UNLOCATED rather than silently dropped.
MANUSCRIPT_NUMBERS = (
    ("populations: low/high-SF units and unit-image pairs",
     "fig4_story_panel_b_values.csv", "n_selected_units / n_selected_unit_image_pairs (unique)",
     "71/7100, 29/2900, 57/977, 22/356"),
    ("panel C/E matched last-bin contrast",
     "fig4_matched_bins_bracket_last_bin_contrast.csv",
     "across_minus_along_percent_point, contrast_p_image_bootstrap_sign, "
     "last_bin_n_selected_units, last_bin_n_selected_unit_image_pairs", None),
    ("panel E last displayed bin (-5.1 pp, p = 0.0004)",
     "fig4_path_bins_last_bin_contrasts.csv",
     "across_minus_along_percent_point + contrast_p_image_bootstrap_sign for the displayed population/bin", None),
    ("panel E first bin (across +4.1% p=0.003, along +2.4% p=0.088)",
     "fig4_path_bins_values.csv",
     "ssi_percent_vs_cell_baseline + population_delta_p_image_bootstrap_sign at component_bin_order min", None),
    ("panel G highest coherence bin (0.155 pp, CI [0.044, 0.265])",
     "fig4_bridge_prediction_by_coherence_summary.csv",
     "observed_minus_rotated + observed_minus_rotated_ci95_low/high at max coherence_bin_order", None),
    ("panel F (11,749 fixation windows, 30 sessions)",
     "fig4_edge_coherence_profiles.csv", "window/session counts", None),
    ("panel A example (0.14 vs 0.10 bits/spike)",
     "fig4_unit_maps_ssi_all_units.csv", "displayed_movie_time_resolved_ssi_bits_per_spike for the example unit", None),
)


def manuscript_numbers(cache_dir: Path) -> int:
    import pandas as pd

    print("=" * 100)
    print(f"MANUSCRIPT NUMBERS  cache={cache_dir}")
    print("=" * 100)
    print("Each row states where the manuscript number is read from. Compare a regenerated")
    print("cache against these to decide whether the manuscript must be updated.\n")

    for label, filename, field_desc, _expected in MANUSCRIPT_NUMBERS:
        path = cache_dir / filename
        print(f"-- {label}")
        print(f"   source: {filename}   field: {field_desc}")
        if not path.exists():
            print("   value : <cache file absent>\n")
            continue
        try:
            if filename == "fig4_story_panel_b_values.csv":
                df = pd.read_csv(path)
                pairs = (df[["sf_group", "relation", "n_selected_units", "n_selected_unit_image_pairs"]]
                         .drop_duplicates().sort_values("n_selected_units", ascending=False))
                for _, r in pairs.iterrows():
                    print(f"   value : {r.sf_group:>12} / {r.relation:<22} "
                          f"{int(r.n_selected_units):>4} units, {int(r.n_selected_unit_image_pairs):>6} pairs")
            elif filename == "fig4_matched_bins_bracket_last_bin_contrast.csv":
                r = pd.read_csv(path).iloc[0]
                print(f"   value : {r.across_minus_along_percent_point:+.3f} pp, "
                      f"p = {r.contrast_p_image_bootstrap_sign:g}, "
                      f"CI [{r.contrast_ci95_low_image_boot:.3f}, {r.contrast_ci95_high_image_boot:.3f}], "
                      f"n = {int(r.last_bin_n_selected_units)} units / "
                      f"{int(r.last_bin_n_selected_unit_image_pairs)} pairs")
            elif filename == "fig4_bridge_prediction_by_coherence_summary.csv":
                df = pd.read_csv(path)
                top = df[df.coherence_bin_order == df.coherence_bin_order.max()]
                for _, r in top.iterrows():
                    print(f"   value : {r.coherence_bin} {r.component_label}: "
                          f"{r.observed_minus_rotated:+.4f} "
                          f"CI [{r.observed_minus_rotated_ci95_low:.4f}, "
                          f"{r.observed_minus_rotated_ci95_high:.4f}]")
            elif filename == "fig4_path_bins_last_bin_contrasts.csv":
                df = pd.read_csv(path)
                for _, r in df.iterrows():
                    print(f"   value : {r.population_key}/{r.metric_family} bin {int(r.component_bin_order)}: "
                          f"{r.across_minus_along_percent_point:+.3f} pp, "
                          f"p = {r.contrast_p_image_bootstrap_sign:g}")
            else:
                df = pd.read_csv(path)
                print(f"   value : <not auto-extracted> ({df.shape[0]} rows x {df.shape[1]} cols)")
        except Exception as exc:  # noqa: BLE001
            print(f"   value : <extraction failed: {type(exc).__name__}: {exc}>")
        print()
    print("=" * 100)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="execute runnable stages into --scratch")
    parser.add_argument("--verify", action="store_true", help="diff staged outputs against --baseline")
    parser.add_argument("--manuscript-numbers", action="store_true", help="report manuscript numbers from a cache dir")
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/fig4_refresh_scratch"))
    parser.add_argument("--baseline", type=Path, default=Path.home() / "fig4_cache_baseline")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--only", type=str, default=None, help="comma-separated stage keys")
    parser.add_argument("--allow-in-place", action="store_true",
                        help="run stages with no --out-dir flag (they write to their default out-dir)")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args(argv)

    only = set(args.only.split(",")) if args.only else None
    if args.manuscript_numbers:
        return manuscript_numbers(args.cache_dir)
    if args.verify:
        return verify(args.scratch, args.baseline, args.rtol, args.atol)
    if args.run:
        return run(args.scratch, only, args.allow_in_place)
    return preflight()


if __name__ == "__main__":
    raise SystemExit(main())
