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

# The six upstream producers. They were never committed to either repo, but they
# do exist on disk -- in two directories that were mode 700, which is why an
# earlier search concluded they were absent. Recovered by copying those
# directories out; the agent transcripts that named them are the provenance.
# Override with FIG4_RECOVERED_ROOT if the copy lives elsewhere.
RECOVERED_ROOT = Path(os.environ.get(
    "FIG4_RECOVERED_ROOT", "/home/ryanress/declan_recovery/VisionCore/declan"))
UPSTREAM_SCRIPT_DIR = RECOVERED_ROOT / "active_sensing_movie_information"
FIG_SSI_SCRIPT_DIR = RECOVERED_ROOT / "fig_ssi"
FIXSTATS_SCRIPT_DIR = RECOVERED_ROOT / "fixation_statistics_by_stimulus"

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
# `outputs/cache/`. They are now declared in `_fig4_paths.REFRESH_ONLY_INPUTS`
# with entries in REFRESH_SOURCES, so this alias exists only so the rest of this
# module keeps reading from one place. They stay out of REQUIRED_INPUTS on
# purpose: the compose path never imports the geometry-story module, so listing
# them there would break rendering a figure that in fact renders.
UNDECLARED_INPUTS = _paths.REFRESH_ONLY_INPUTS


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
    #: Some producers use package-relative imports and must be run as a module
    #: (`python -m pkg.mod`) rather than as a script. When set, this is the
    #: dotted module name, and `cwd` is the directory to run it from.
    module: str | None = None
    cwd: Path | None = None
    #: Extra CLI arguments. "{prev_out}" expands to the scratch out-dir of the
    #: first entry in `needs`, so a stage can consume an upstream stage's
    #: freshly regenerated output rather than a pre-existing copy.
    extra_args: tuple[str, ...] = ()
    #: Directory to prepend to PYTHONPATH so `module` resolves. Defaults to the
    #: fig4 dir; recovered upstream packages live elsewhere.
    pythonpath: Path | None = None
    #: Directory whose contents are copied into the scratch out-dir before the
    #: stage runs. For producers whose out-dir is also an input root, so they can
    #: read-modify-write in scratch instead of over the original.
    seed_from: Path | None = None
    needs: tuple[str, ...] = ()
    note: str = ""

    def missing_inputs(self) -> list[Path]:
        return [p for p in self.inputs if not p.exists()]

    def status(self) -> tuple[str, str]:
        if self.script is None:
            return "NO-SCRIPT", f"producer '{self.upstream_name}' exists in no tree or git history"
        if not self.script.exists():
            return "NO-SCRIPT", f"{self.script} not found"
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

# Panel H's radius sweep. `summarize_backimage_patch_radius_sensitivity` reads
# one pole table and one sf-scaling table per radius; both sets come from
# screening scripts that the declared graph did not mention.
_PATCH_RADIUS_ROOT = UPSTREAM_FIXSTATS / "backimage_patch_radius_sensitivity_v1"
_PATCH_RADIUS_LABELS = ("r0p25", "r0p5", "r1p0")
_PATCH_RADIUS_POLE_INPUTS = tuple(
    _PATCH_RADIUS_ROOT / f"local_feature_poles_{label}" / "pole_eye_metric_high_low_contrasts.csv"
    for label in _PATCH_RADIUS_LABELS
)
_PATCH_RADIUS_SF_INPUTS = tuple(
    _PATCH_RADIUS_ROOT / f"sf_scaling_{label}" / "sf_controlled_slope_summary.csv"
    for label in _PATCH_RADIUS_LABELS
)

# The RR100 branch. These two paths are NOT the script defaults and NOT the
# similarly-named run dirs that sit beside them: they are the values recorded in
# `cache_identity_json` inside the shipped `fig4_unit_maps.npz`, which is the
# only statement of what actually produced the cache. The plausible-by-name
# alternative (`..._sf_contour_alignment_long_axis30_...`) postdates the cache.
UNIT_MAPS_SOURCE_RUN_DIR = (
    UPSTREAM_MOVIE_INFO / "backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1"
)
UNIT_MAPS_AXIS_RUN_DIR = (
    UPSTREAM_FIXSTATS
    / "backimage_axis_conditioned_matched_static_percandidate_gpu1_n128_c4_k16_scales_0p5_1_2_bconsistent_v1"
)
UNIT_MAPS_RUN_DIR = UPSTREAM_MOVIE_INFO / "backimage_rr100_instantaneous_unit_maps_latest_v1"
SF_TUNING_DIR = (
    UPSTREAM_MOVIE_INFO / "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1"
)

# The shard tree `merge_backimage_real_trace_ssi_matrix_shards.py` merges.
SSI_SHARDS_DIR = (
    UPSTREAM_MOVIE_INFO
    / "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1"
    / "shards"
)

# Dataset configs pinned to the revision whose per-session `cids` match the
# checkpoint's readout shapes. Without this the twin model refuses to load:
# `MultiDatasetModel` sizes each readout from `len(cids)` read live off disk, the
# checkpoint stores no cids of its own, and the configs it points at have since
# drifted. Verified: readouts load at strict=True and the canonical selection
# comes to 756 units, the number `CanonicalTwinScorer` documents.
PINNED_DATASET_CONFIGS_REV = "e6c85ae"
PINNED_DATASET_CONFIGS = Path(os.environ.get(
    "FIG4_DATASET_CONFIGS",
    str(FIG4_DIR / "upstream" / "dataset_configs" / "multi_basic_120_long.yaml"),
))


STAGES: tuple[Stage, ...] = (
    # -- Tier 0: producers that exist nowhere ------------------------------
    Stage(
        key="instantaneous_unit_maps",
        script=UPSTREAM_SCRIPT_DIR / "plot_backimage_rr100_instantaneous_unit_maps.py",
        upstream_name="plot_backimage_rr100_instantaneous_unit_maps.py",
        inputs=(
            UNIT_MAPS_SOURCE_RUN_DIR / "cache" / "backimage_contour_axis_rr100_spatial_ssi_cache.npz",
            UNIT_MAPS_AXIS_RUN_DIR,
        ),
        produces={
            # Real emitted filenames. The previous four names ("unit_maps.npz"
            # etc.) match nothing this producer writes, so the stage would have
            # staged zero files and still reported success -- the same silent
            # failure the bracket stage had.
            "cache/backimage_rr100_instantaneous_unit_maps.npz": _paths.UNIT_MAPS_NPZ,
            "cache/selected_patch.npy": _paths.UNIT_MAPS_SELECTED_PATCH_NPY,
            "displayed_movie_instantaneous_ssi_all_units.csv": _paths.UNIT_MAPS_SSI_ALL_UNITS_CSV,
            "orientation_tuning_groups.csv": _paths.UNIT_MAPS_ORIENTATION_GROUPS_CSV,
        },
        note=(
            "RR100 movie run over the recordings; panel A/C instantaneous unit maps. "
            "Needs the twin model, so it needs FIG4_DATASET_CONFIGS pinned (see "
            "PINNED_DATASET_CONFIGS). Its --source-run-dir and --axis-run-dir come "
            "from cache_identity_json inside the shipped fig4_unit_maps.npz, not from "
            "the defaults in the script."
        ),
        out_dir_flag="--out-dir",
    ),
    Stage(
        key="merge_ssi_shards",
        script=UPSTREAM_SCRIPT_DIR / "merge_backimage_real_trace_ssi_matrix_shards.py",
        upstream_name="merge_backimage_real_trace_ssi_matrix_shards.py",
        inputs=(SSI_SHARDS_DIR,),
        produces={
            "image_feature_table.csv": _paths.IMAGE_FEATURE_TABLE_CSV,
            "trace_xy.npy": _paths.TRACE_XY_NPY,
        },
        note=(
            "Produces the merged trace bank at TRACE_BANK_MERGED_DIR; the two cache files "
            "are copies of members of that bank. Linchpin: every geometry-story stage needs it."
        ),
        out_dir_flag="--out-dir",
    ),
    Stage(
        key="phase1_phase2",
        script=UPSTREAM_SCRIPT_DIR / "analyze_backimage_real_trace_ssi_matrix_phase1_phase2.py",
        upstream_name="analyze_backimage_real_trace_ssi_matrix_phase1_phase2.py",
        inputs=_BANK_INPUTS,
        produces={"phase1_movie_analysis_table.csv": _paths.PHASE1_MOVIE_ANALYSIS_TABLE_CSV},
        needs=("merge_ssi_shards",),
        note=(
            "Does NOT produce trace_component_movie_metrics.csv -- 10+ of that file's "
            "27 columns never appear in this script's output. See matched_trace_components."
        ),
        out_dir_flag="--out-dir",
    ),
    Stage(
        key="matched_trace_components",
        script=UPSTREAM_SCRIPT_DIR / "analyze_backimage_contour_matched_trace_components.py",
        upstream_name="analyze_backimage_contour_matched_trace_components.py",
        inputs=_BANK_INPUTS,
        produces={
            "phase2_contour_relative_trace_component_movie_metrics.csv":
                _paths.TRACE_COMPONENT_MOVIE_METRICS_CSV,
        },
        needs=("merge_ssi_shards",),
        note=(
            "True producer of fig4_trace_component_movie_metrics.csv, confirmed by md5 "
            "against the recovered output. Writes into "
            "merged/phase1_phase2_conditioning_v1/trace_component_conditioning_v1/."
        ),
        out_dir_flag="--out-dir",
    ),
    Stage(
        key="sf_group_ssi_modulation",
        script=UPSTREAM_SCRIPT_DIR / "plot_backimage_rr100_sf_group_ssi_modulation.py",
        upstream_name="plot_backimage_rr100_sf_group_ssi_modulation.py",
        inputs=(
            SF_TUNING_DIR / "frequency_tuning_summary.csv",
            SF_TUNING_DIR / "frequency_tuning_grouped.csv",
            UNIT_MAPS_RUN_DIR / "displayed_movie_instantaneous_ssi_all_units.csv",
        ),
        needs=("instantaneous_unit_maps",),
        produces={
            "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv":
                _paths.SF_TUNING_UNIT_GROUPS_CSV,
        },
        note=(
            "True producer of fig4_sf_tuning_unit_groups.csv, confirmed by md5. The "
            "filename prefix is the --sf-metric value. run_backimage_rr100_frequency_"
            "tuning_probe.py writes to an unrelated out-dir and never emits this file. "
            "Self-documenting output: frequency_tuning_contract and sf_split_metric_column "
            "record the formula, so the method is reconstructable from the cache if the script is not."
        ),
        out_dir_flag="--out-dir",
    ),
    Stage(
        key="schematic_final_maps",
        script=FIG_SSI_SCRIPT_DIR / "compute_schematic_rr100_final_maps.py",
        upstream_name="compute_schematic_rr100_final_maps.py",
        inputs=(UNIT_MAPS_RUN_DIR,),
        produces={
            # Real emitted names/locations, md5-confirmed against the recovered
            # run dir. This stage emits only two of the three schematic caches;
            # trace_center40 comes from make_ssi_contour_schematic (below).
            "cache/schematic_rr100_final_maps.npz": _paths.SCHEMATIC_FINAL_MAPS_NPZ,
            "schematic_rr100_final_map_unit_metrics.csv": _paths.SCHEMATIC_FINAL_MAP_UNIT_METRICS_CSV,
        },
        needs=("instantaneous_unit_maps",),
        note=(
            "Reads only the instantaneous unit-maps run dir (its RUN_DIR), so it is the "
            "shallowest stage that exercises the twin model end to end. Defaults to cuda:1."
        ),
        out_dir_flag="--out-dir",
    ),
    Stage(
        key="contour_schematic_trace",
        script=FIG_SSI_SCRIPT_DIR / "make_ssi_contour_schematic.py",
        upstream_name="make_ssi_contour_schematic.py",
        produces={
            "schematic_crop_real_backimage_trace_center40.csv":
                _paths.SCHEMATIC_TRACE_CENTER40_CSV,
        },
        default_out_dir=ROOT / "outputs" / "fig_ssi" / "trace_provenance",
        note=(
            "True producer of fig4_schematic_trace_center40.csv, confirmed by md5. "
            "compute_schematic_rr100_final_maps.py reads this trace rather than "
            "writing it, so the previous mapping credited a consumer."
        ),
        out_dir_flag=None,
    ),
    Stage(
        key="schematic_stimulus_payload",
        script=REFRESH_DIR / "build_schematic_stimulus_cache.py",
        inputs=(_paths.IMAGE_FEATURE_TABLE_CSV, _paths.SCHEMATIC_TRACE_CENTER40_CSV),
        produces={
            "fig4_schematic_stimulus_payload.npz": _paths.SCHEMATIC_STIMULUS_PAYLOAD_NPZ,
        },
        default_out_dir=CACHE_DIR,
        out_dir_flag="--out-dir",
        needs=("merge_ssi_shards", "contour_schematic_trace"),
        note=(
            "Cache-only boundary for panel A/C. Requires DataYatesV1/raw BackImage "
            "data when run, but compose reads only the staged npz."
        ),
    ),
    Stage(
        key="matched_bins_bracket",
        script=UPSTREAM_SCRIPT_DIR / "make_backimage_panel_c_sf05_match15_matched_bins_bracket.py",
        upstream_name="make_backimage_panel_c_sf05_match15_matched_bins_bracket.py",
        default_out_dir=(UPSTREAM_MOVIE_INFO
                         / "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1"
                         / "merged" / "phase1_phase2_conditioning_v1" / "plot_collections"),
        pythonpath=RECOVERED_ROOT.parent,
        inputs=_BANK_INPUTS,
        produces={
            "backimage_real_trace_panel_c_aligned_sf_ge_0p5_match15_matched_bins_bracket_values.csv": _paths.MATCHED_BINS_BRACKET_VALUES_CSV,
            "backimage_real_trace_panel_c_aligned_sf_ge_0p5_match15_matched_bins_bracket_last_bin_contrast.csv": _paths.MATCHED_BINS_BRACKET_LAST_BIN_CONTRAST_CSV,
            "backimage_real_trace_panel_c_aligned_sf_ge_0p5_match15_matched_bins_bracket_summary.json": _paths.MATCHED_BINS_BRACKET_SUMMARY_JSON,
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
        produces={},
        default_out_dir=UPSTREAM_FIXSTATS / "backimage_contour_motion_component_plots_v1",
        out_dir_flag="--out-dir",
        module="fixation_stats.plot_backimage_contour_motion_components",
        note=("Emits contour_motion_component_windows.csv only. REFRESH_SOURCES credits it "
              "with the two panel F edge-coherence caches, but it does not write them -- "
              "edge_coherence_followups does."),
    ),
    Stage(
        key="local_feature_poles",
        script=FIXSTATS_SCRIPT_DIR / "screen_backimage_local_feature_eye_metric_poles.py",
        upstream_name="screen_backimage_local_feature_eye_metric_poles.py",
        produces={},
        default_out_dir=_PATCH_RADIUS_ROOT,
        out_dir_flag="--out-dir",
        note=(
            "Undeclared panel H prerequisite. Run once per radius into "
            "local_feature_poles_{r0p25,r0p5,r1p0}/. Produces no fig4_* cache directly; "
            "panel H reads its pole_eye_metric_high_low_contrasts.csv."
        ),
    ),
    Stage(
        key="sf_eye_metric_scaling",
        script=FIXSTATS_SCRIPT_DIR / "screen_backimage_spatial_frequency_eye_metric_scaling.py",
        upstream_name="screen_backimage_spatial_frequency_eye_metric_scaling.py",
        produces={},
        default_out_dir=_PATCH_RADIUS_ROOT,
        out_dir_flag="--out-dir",
        note=(
            "Second undeclared panel H prerequisite. Run once per radius into "
            "sf_scaling_{r0p25,r0p5,r1p0}/. Panel H reads its sf_controlled_slope_summary.csv."
        ),
    ),
    Stage(
        key="patch_radius_sensitivity",
        script=REFRESH_DIR / "summarize_backimage_patch_radius_sensitivity.py",
        # The pole and sf-scaling tables are real inputs. They were previously
        # undeclared and happened to be satisfied by `seed_from`, so the stage
        # passed while two of its prerequisites were invisible. Declared here so
        # a missing one blocks the stage instead of quietly seeding a pass.
        inputs=(WINDOW_FEATURES_CSV,) + _PATCH_RADIUS_POLE_INPUTS + _PATCH_RADIUS_SF_INPUTS,
        produces={
            "patch_radius_alignment_slope_coherence_gt0p3.csv": _paths.PATCH_RADIUS_ALIGNMENT_SLOPE_CSV,
            "patch_radius_alignment_by_coherence.pdf": _paths.PATCH_RADIUS_ALIGNMENT_BY_COHERENCE_PDF,
        },
        default_out_dir=_PATCH_RADIUS_ROOT,
        out_dir_flag="--root",
        seed_from=_PATCH_RADIUS_ROOT,
        needs=("local_feature_poles", "sf_eye_metric_scaling"),
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
    Stage(
        key="edge_coherence_followups",
        script=FIXSTATS_SCRIPT_DIR / "generate_backimage_contour_position_spread_followups.py",
        upstream_name="generate_backimage_contour_position_spread_followups.py",
        produces={
            "b_position_spread_unwrapped_profiles_by_edge_coherence.csv": _paths.EDGE_COHERENCE_PROFILES_CSV,
            "b_position_spread_random_orientation_baseline_by_edge_coherence.csv":
                _paths.EDGE_COHERENCE_RANDOM_BASELINE_CSV,
        },
        default_out_dir=UPSTREAM_FIXSTATS / "backimage_contour_motion_component_plots_v1",
        out_dir_flag="--out-dir",
        extra_args=("--input-windows", "{prev_out}/contour_motion_component_windows.csv"),
        module="fixation_statistics_by_stimulus.generate_backimage_contour_position_spread_followups",
        pythonpath=RECOVERED_ROOT,
        needs=("contour_motion_components",),
        note="Panel F. The producer REFRESH_SOURCES omits entirely.",
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

# Hand-tuned, declared not regenerated. Listed so the report accounts for all
# required inputs rather than silently treating layout-only artifacts as missing
# producer coverage.
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

    print("\n-- refresh-only caches (declared; not required to compose) " + "-" * 39)
    for path in UNDECLARED_INPUTS:
        source = _paths.REFRESH_SOURCES.get(path, "unknown")
        print(f"  [{'present' if path.exists() else 'ABSENT ':7}] {path.name}")
        print(f"            read by refresh/_fig4_geometry_story.py  <- {source}")

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
        if stage.seed_from is not None:
            if not stage.seed_from.exists():
                results.append({"stage": stage.key, "result": "BLOCKED",
                                "detail": f"seed dir {_short(stage.seed_from)} absent"})
                print(f"[skip BLOCKED   ] {stage.key}: seed dir {_short(stage.seed_from)} absent")
                continue
            shutil.copytree(stage.seed_from, out_dir, dirs_exist_ok=True)
            # The seed root generally already holds this stage's own outputs.
            # Delete them, or a stage that silently fails to rewrite a file
            # would have its seeded copy staged and verify as a false PASS.
            cleared = 0
            for out_name in stage.produces:
                seeded = out_dir / out_name
                if seeded.exists():
                    seeded.unlink()
                    cleared += 1
            print(f"[seed           ] {stage.key}: from {_short(stage.seed_from)}"
                  f" ({cleared} pre-existing output(s) cleared)")
        cmd = ([sys.executable, "-m", stage.module] if stage.module
               else [sys.executable, str(stage.script)])
        if stage.extra_args:
            prev_out = str(scratch / stage.needs[0]) if stage.needs else ""
            cmd += [a.replace("{prev_out}", prev_out) for a in stage.extra_args]
        if stage.out_dir_flag:
            cmd += [stage.out_dir_flag, str(out_dir)]
            effective_out = out_dir
        else:
            effective_out = stage.default_out_dir
            if effective_out is None:
                results.append({"stage": stage.key, "result": "NO-OUT-DIR",
                                "detail": "no --out-dir flag and no declared default_out_dir; "
                                          "cannot locate its outputs to stage"})
                print(f"[NO-OUT-DIR     ] {stage.key}: cannot locate outputs")
                continue
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
        # Module-invoked stages import from the fig4 package dir but resolve
        # their data paths relative to the repo root.
        if stage.module:
            env["PYTHONPATH"] = os.pathsep.join(
                [str(stage.pythonpath or FIG4_DIR)]
                + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        proc = subprocess.run(cmd, cwd=str(stage.cwd or ROOT), env=env, capture_output=True, text=True)
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

    # Everything the figure requires, plus anything the run staged that is not
    # a required input (the bridge rotation null, for one) -- a staged file left
    # uncompared is exactly the kind of silent gap this step exists to close.
    required = list(_paths.REQUIRED_INPUTS)
    required_names = {p.name for p in required}
    extra = sorted(p for p in staged_dir.glob("*") if p.is_file() and p.name not in required_names) \
        if staged_dir.exists() else []

    for cache_path in required + extra:
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
            # Column *order* is not a reproducibility failure -- the figure reads
            # by name. Reorder and say so; only a set difference is a real fail.
            note = ""
            superset: list[str] = []
            if list(a.columns) != list(b.columns):
                only_new = set(a.columns) - set(b.columns)
                only_old = set(b.columns) - set(a.columns)
                if only_old:
                    return "FAIL", (f"columns missing from regenerated file: {sorted(only_old)}"
                                    + (f" (also new: {sorted(only_new)})" if only_new else ""))
                if only_new:
                    superset = sorted(only_new)
                a = a[list(b.columns)]
                note = ", column order differs" if not superset else ""
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
            detail = (f"{a.shape[0]}x{a.shape[1]}, worst rel dev {worst:.2e}"
                      + (f" ({worst_col})" if worst_col else "") + note)
            if superset:
                return "PASS-STALE-CACHE", detail + f"; regenerated adds {superset}"
            return "PASS", detail

        if suffix == ".npy":
            a, b = np.load(new, allow_pickle=False), np.load(old, allow_pickle=False)
            if a.shape != b.shape:
                return "FAIL", f"shape {a.shape} vs {b.shape}"
            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                return "FAIL", f"max abs dev {float(np.nanmax(np.abs(a - b))):.3g}"
            return "PASS", f"shape {a.shape}"

        if suffix == ".npz":
            a, b = np.load(new, allow_pickle=True), np.load(old, allow_pickle=True)
            if set(a.files) != set(b.files):
                return "FAIL", f"keys differ: {set(a.files) ^ set(b.files)}"
            for key in b.files:
                x, y = a[key], b[key]
                if x.shape != y.shape:
                    return "FAIL", f"{key}: shape {x.shape} vs {y.shape}"
                if x.dtype.kind == "O":
                    if not np.array_equal(x, y):
                        return "FAIL", f"{key}: object-array values differ"
                elif x.dtype.kind in "fc":
                    if not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
                        return "FAIL", f"{key}: max abs dev {float(np.nanmax(np.abs(x - y))):.3g}"
                elif not np.array_equal(x, y):
                    return "FAIL", f"{key}: values differ"
            return "PASS", f"{len(b.files)} array(s)"

        if suffix == ".json":
            na, nb = json.loads(new.read_text()), json.loads(old.read_text())
            bad = _json_diff(na, nb, rtol, atol, "")
            if not bad:
                return "PASS", ""
            stripped = _json_diff(_drop_provenance(na), _drop_provenance(nb), rtol, atol, "")
            if not stripped:
                return "PASS-PROVENANCE", f"differs only in recorded paths ({bad.split(':')[0]})"
            return "FAIL", stripped

        if suffix == ".pdf":
            # Vector output is not byte-reproducible (timestamps, font subset ids).
            return "SKIP-BINARY", f"pdf, {new.stat().st_size} vs {old.stat().st_size} bytes"

        return "SKIP-BINARY", f"no comparator for {suffix}"
    except Exception as exc:  # noqa: BLE001 - the comparison itself failing is a result
        return "FAIL", f"comparison raised {type(exc).__name__}: {exc}"


#: JSON keys that record where an analysis ran rather than what it found.
PROVENANCE_KEYS = frozenset({
    "matrix_dir", "out_dir", "root", "input_windows", "cache_dir", "source_dir",
    "generated_at", "timestamp", "command", "argv", "hostname", "script",
})


def _drop_provenance(obj):
    """Normalise recorded locations without discarding what was recorded.

    Path-valued strings are reduced to their basename, so a run from a different
    directory still has to name the same files -- only the prefix is forgiven.
    Keys that record run identity rather than result are dropped outright.
    """
    if isinstance(obj, dict):
        return {k: _drop_provenance(v) for k, v in obj.items() if k not in PROVENANCE_KEYS}
    if isinstance(obj, list):
        return [_drop_provenance(v) for v in obj]
    if isinstance(obj, str) and "/" in obj and not obj.strip().startswith("http"):
        return obj.rstrip("/").rsplit("/", 1)[-1]
    return obj


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
