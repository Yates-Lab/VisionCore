"""Canonical paths for figure 4.

Every cached input the compose path reads lives flat under `CACHE_DIR` as
`fig4_*`, matching fig3's convention. The names describe what the file holds
rather than which analysis run produced it -- the run identity is provenance,
recorded in the manifest, not something a reader should have to decode from a
path like `backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_
n100x1000_v1/merged/phase1_phase2_conditioning_v1/...`.

Outputs go to `FIGURES_DIR/fig4` and `STATS_DIR/fig4`; both are gitignored and
created at import by `VisionCore.paths`.

`REFRESH_SOURCES` maps each cached input back to the script that regenerates
it, so the manifest can state provenance and the preflight can tell you what to
run when something is missing.
"""
from __future__ import annotations

from VisionCore.paths import CACHE_DIR, FIGURES_DIR, STATS_DIR, VISIONCORE_ROOT

FIG4_DIR = VISIONCORE_ROOT / "paper" / "fig4"
REFERENCE_DIR = FIG4_DIR / "reference"
REFRESH_DIR = FIG4_DIR / "refresh"

# Outputs -----------------------------------------------------------------
FIG_DIR = FIGURES_DIR / "fig4"
STAT_DIR = STATS_DIR / "fig4"
PANELS_DIR = FIG_DIR / "panels"

# The Illustrator artwork the measured layout boxes came from, and the source
# the panel A network icon is extracted from on a cache miss.
LAYOUT_REFERENCE_PDF = REFERENCE_DIR / "ssi_figure_v2_3.pdf"

# Refresh-only inputs -----------------------------------------------------
# The merged real-trace SSI matrix. Only the refresh scripts read it, and it is
# far too large to keep staged, so it is deliberately absent: a refresh run has
# to stage the merged bank here first, and fails loudly if it has not.
TRACE_BANK_MERGED_DIR = CACHE_DIR / "fig4_trace_bank_merged"

# Cached inputs -----------------------------------------------------------
# Instantaneous unit maps (the RR100 movie run).
UNIT_MAPS_NPZ = CACHE_DIR / "fig4_unit_maps.npz"
UNIT_MAPS_SELECTED_PATCH_NPY = CACHE_DIR / "fig4_unit_maps_selected_patch.npy"
UNIT_MAPS_SSI_ALL_UNITS_CSV = CACHE_DIR / "fig4_unit_maps_ssi_all_units.csv"
UNIT_MAPS_ORIENTATION_GROUPS_CSV = CACHE_DIR / "fig4_unit_maps_orientation_groups.csv"

# Real-trace bank.
IMAGE_FEATURE_TABLE_CSV = CACHE_DIR / "fig4_image_feature_table.csv"
TRACE_XY_NPY = CACHE_DIR / "fig4_trace_xy.npy"
TRACE_COMPONENT_MOVIE_METRICS_CSV = CACHE_DIR / "fig4_trace_component_movie_metrics.csv"

# Spatial-frequency tuning groups.
SF_TUNING_UNIT_GROUPS_CSV = CACHE_DIR / "fig4_sf_tuning_unit_groups.csv"

# Panel A schematic endpoint maps.
SCHEMATIC_FINAL_MAPS_NPZ = CACHE_DIR / "fig4_schematic_final_maps.npz"
SCHEMATIC_FINAL_MAP_UNIT_METRICS_CSV = CACHE_DIR / "fig4_schematic_final_map_unit_metrics.csv"
SCHEMATIC_TRACE_CENTER40_CSV = CACHE_DIR / "fig4_schematic_trace_center40.csv"

# Panel A network icon (extracted from the Illustrator reference).
PANEL_A_NETWORK_ICON_PDF = CACHE_DIR / "fig4_panel_a_network_icon.pdf"
PANEL_A_NETWORK_ICON_PROVENANCE_JSON = CACHE_DIR / "fig4_panel_a_network_icon_provenance.json"
PANEL_A_LAYOUT_OVERRIDES_JSON = CACHE_DIR / "fig4_panel_a_layout_overrides.json"

# Panel C contour-relative stimulus.
PANEL_D_LAYOUT_OVERRIDES_JSON = CACHE_DIR / "fig4_panel_d_layout_overrides.json"
COHERENCE_GALLERY_NPZ = CACHE_DIR / "fig4_coherence_gallery.npz"

# Path-length bins (panels B/D/E).
PATH_BINS_VALUES_CSV = CACHE_DIR / "fig4_path_bins_values.csv"
PATH_BINS_LAST_BIN_CONTRASTS_CSV = CACHE_DIR / "fig4_path_bins_last_bin_contrasts.csv"
PATH_BINS_TRACE_BANK_REFERENCE_CSV = CACHE_DIR / "fig4_path_bins_trace_bank_reference.csv"
PATH_BINS_POPULATIONS_CSV = CACHE_DIR / "fig4_path_bins_populations.csv"
BEHAVIOR_PATH_WINDOWS_CSV = CACHE_DIR / "fig4_behavior_path_windows.csv"

# Behaviour/model bridge (panel G).
BRIDGE_PREDICTION_BY_COHERENCE_SUMMARY_CSV = CACHE_DIR / "fig4_bridge_prediction_by_coherence_summary.csv"
BRIDGE_MATCH_NULL_SUMMARY_CSV = CACHE_DIR / "fig4_bridge_match_null_summary.csv"

# Edge coherence (panel F) and patch radius (panel H).
EDGE_COHERENCE_PROFILES_CSV = CACHE_DIR / "fig4_edge_coherence_profiles.csv"
EDGE_COHERENCE_RANDOM_BASELINE_CSV = CACHE_DIR / "fig4_edge_coherence_random_baseline.csv"
PATCH_RADIUS_ALIGNMENT_SLOPE_CSV = CACHE_DIR / "fig4_patch_radius_alignment_slope.csv"
PATCH_RADIUS_ALIGNMENT_BY_COHERENCE_PDF = CACHE_DIR / "fig4_patch_radius_alignment_by_coherence.pdf"

# Story/bracket collections.
STORY_COMPONENT_VALUES_CSV = CACHE_DIR / "fig4_story_component_values.csv"
STORY_COMPONENT_SELECTION_SUMMARY_CSV = CACHE_DIR / "fig4_story_component_selection_summary.csv"
STORY_COMPONENT_SUMMARY_JSON = CACHE_DIR / "fig4_story_component_summary.json"
STORY_PANEL_B_VALUES_CSV = CACHE_DIR / "fig4_story_panel_b_values.csv"
STORY_PANEL_B_SELECTION_SUMMARY_CSV = CACHE_DIR / "fig4_story_panel_b_selection_summary.csv"
STORY_PANEL_B_SUMMARY_JSON = CACHE_DIR / "fig4_story_panel_b_summary.json"
MATCHED_BINS_BRACKET_VALUES_CSV = CACHE_DIR / "fig4_matched_bins_bracket_values.csv"
MATCHED_BINS_BRACKET_LAST_BIN_CONTRAST_CSV = CACHE_DIR / "fig4_matched_bins_bracket_last_bin_contrast.csv"
MATCHED_BINS_BRACKET_SUMMARY_JSON = CACHE_DIR / "fig4_matched_bins_bracket_summary.json"


# Which refresh script regenerates each cached input. Values name scripts in
# `refresh/` where fig4 owns the step, or the originating analysis where it
# does not -- several caches come from runs that live outside this repo and
# need the raw recordings, so the honest answer is a name, not a path.
REFRESH_SOURCES = {
    UNIT_MAPS_NPZ: "run_backimage_contour_axis_rr100_spatial_ssi.py (upstream)",
    UNIT_MAPS_SELECTED_PATCH_NPY: "run_backimage_contour_axis_rr100_spatial_ssi.py (upstream)",
    UNIT_MAPS_SSI_ALL_UNITS_CSV: "run_backimage_contour_axis_rr100_spatial_ssi.py (upstream)",
    UNIT_MAPS_ORIENTATION_GROUPS_CSV: "run_backimage_contour_axis_rr100_spatial_ssi.py (upstream)",
    IMAGE_FEATURE_TABLE_CSV: "merge_backimage_real_trace_ssi_matrix_shards.py (upstream)",
    TRACE_XY_NPY: "merge_backimage_real_trace_ssi_matrix_shards.py (upstream)",
    TRACE_COMPONENT_MOVIE_METRICS_CSV: "analyze_backimage_real_trace_ssi_matrix_phase1_phase2.py (upstream)",
    SF_TUNING_UNIT_GROUPS_CSV: "run_backimage_rr100_frequency_tuning_probe.py (upstream)",
    SCHEMATIC_FINAL_MAPS_NPZ: "compute_schematic_rr100_final_maps.py (upstream)",
    SCHEMATIC_FINAL_MAP_UNIT_METRICS_CSV: "compute_schematic_rr100_final_maps.py (upstream)",
    SCHEMATIC_TRACE_CENTER40_CSV: "compute_schematic_rr100_final_maps.py (upstream)",
    PANEL_A_NETWORK_ICON_PDF: "_fig4_network_icon.py",
    PANEL_A_NETWORK_ICON_PROVENANCE_JSON: "_fig4_network_icon.py",
    PANEL_A_LAYOUT_OVERRIDES_JSON: "hand-tuned layout overrides (tracked provenance, not regenerated)",
    PANEL_D_LAYOUT_OVERRIDES_JSON: "hand-tuned layout overrides (tracked provenance, not regenerated)",
    COHERENCE_GALLERY_NPZ: "refresh/build_coherence_gallery_cache.py",
    PATH_BINS_VALUES_CSV: "refresh/panel_g_alternative_x_axes_diagnostic.py",
    PATH_BINS_LAST_BIN_CONTRASTS_CSV: "refresh/panel_g_alternative_x_axes_diagnostic.py",
    PATH_BINS_TRACE_BANK_REFERENCE_CSV: "refresh/panel_g_alternative_x_axes_diagnostic.py",
    PATH_BINS_POPULATIONS_CSV: "refresh/panel_g_alternative_x_axes_diagnostic.py",
    BEHAVIOR_PATH_WINDOWS_CSV: "refresh/behavior_component_path_by_coherence.py",
    BRIDGE_PREDICTION_BY_COHERENCE_SUMMARY_CSV: "refresh/_fig4_bridge_by_coherence.py",
    BRIDGE_MATCH_NULL_SUMMARY_CSV: "refresh/_fig4_bridge_rotation_null.py",
    EDGE_COHERENCE_PROFILES_CSV: "fixation_stats/plot_backimage_contour_motion_components.py",
    EDGE_COHERENCE_RANDOM_BASELINE_CSV: "fixation_stats/plot_backimage_contour_motion_components.py",
    PATCH_RADIUS_ALIGNMENT_SLOPE_CSV: "refresh/summarize_backimage_patch_radius_sensitivity.py",
    PATCH_RADIUS_ALIGNMENT_BY_COHERENCE_PDF: "refresh/summarize_backimage_patch_radius_sensitivity.py",
    STORY_COMPONENT_VALUES_CSV: "refresh/_fig4_geometry_story_cde8bins.py",
    STORY_PANEL_B_VALUES_CSV: "refresh/_fig4_orientation_match_sf05.py",
    MATCHED_BINS_BRACKET_VALUES_CSV: "make_backimage_panel_c_sf05_match15_matched_bins_bracket.py (upstream)",
    MATCHED_BINS_BRACKET_LAST_BIN_CONTRAST_CSV: "make_backimage_panel_c_sf05_match15_matched_bins_bracket.py (upstream)",
    MATCHED_BINS_BRACKET_SUMMARY_JSON: "make_backimage_panel_c_sf05_match15_matched_bins_bracket.py (upstream)",
}

# The inputs the compose path actually opens. The preflight checks these; the
# rest of the mapping is declared by modules but only read on refresh paths.
REQUIRED_INPUTS = (
    UNIT_MAPS_NPZ,
    UNIT_MAPS_SELECTED_PATCH_NPY,
    UNIT_MAPS_SSI_ALL_UNITS_CSV,
    UNIT_MAPS_ORIENTATION_GROUPS_CSV,
    IMAGE_FEATURE_TABLE_CSV,
    TRACE_XY_NPY,
    TRACE_COMPONENT_MOVIE_METRICS_CSV,
    SF_TUNING_UNIT_GROUPS_CSV,
    SCHEMATIC_FINAL_MAPS_NPZ,
    SCHEMATIC_FINAL_MAP_UNIT_METRICS_CSV,
    SCHEMATIC_TRACE_CENTER40_CSV,
    PANEL_A_NETWORK_ICON_PDF,
    PANEL_A_LAYOUT_OVERRIDES_JSON,
    PANEL_D_LAYOUT_OVERRIDES_JSON,
    COHERENCE_GALLERY_NPZ,
    PATH_BINS_VALUES_CSV,
    PATH_BINS_LAST_BIN_CONTRASTS_CSV,
    PATH_BINS_TRACE_BANK_REFERENCE_CSV,
    PATH_BINS_POPULATIONS_CSV,
    BRIDGE_PREDICTION_BY_COHERENCE_SUMMARY_CSV,
    EDGE_COHERENCE_PROFILES_CSV,
    EDGE_COHERENCE_RANDOM_BASELINE_CSV,
    PATCH_RADIUS_ALIGNMENT_SLOPE_CSV,
)


def missing_inputs(required=REQUIRED_INPUTS):
    """Every required input that is absent, so a caller can report them all at
    once instead of failing on whichever happens to be read first."""
    return [path for path in required if not path.exists()]


class Fig4MissingInput(RuntimeError):
    """A cached input the figure needs is not present.

    This exists because the failure mode it replaces was silent. Panel A used
    to fall back to a schematic drawn from synthetic data when its real payload
    was missing: the panel still rendered, with plausible geometry and correct
    SSI values, and nothing in the output said the data behind it was gone.
    A figure that is quietly wrong is worse than one that fails.
    """


# Degraded rendering is opt-in. `compose(allow_missing=True)` and the
# `--allow-missing` flag set this, so a missing cache draws a visible in-figure
# placeholder instead of raising -- useful when iterating on layout without a
# full cache set, never the default.
ALLOW_MISSING = False


def require(path, what: str):
    """Return `path` if it exists; otherwise raise unless degraded rendering is
    enabled, in which case return None and let the caller draw a placeholder."""
    if path.exists():
        return path
    if ALLOW_MISSING:
        return None
    source = REFRESH_SOURCES.get(path, "unknown")
    raise Fig4MissingInput(
        f"{what} is missing:\n"
        f"  expected: {path}\n"
        f"  regenerate with: {source}\n"
        f"Pass --allow-missing to render a placeholder panel instead."
    )


def check_required_inputs():
    """Preflight: report every missing input at once rather than dying on the
    first one read, which tells you nothing about how much else is absent."""
    missing = missing_inputs()
    if not missing:
        return
    lines = [f"  {path.name}  <- {REFRESH_SOURCES.get(path, 'unknown')}" for path in missing]
    message = (
        f"{len(missing)} of {len(REQUIRED_INPUTS)} required fig4 inputs are missing "
        f"from {CACHE_DIR}:\n" + "\n".join(lines)
    )
    if ALLOW_MISSING:
        print(f"WARNING: {message}\nRendering placeholders for the affected panels.")
        return
    raise Fig4MissingInput(message + "\nPass --allow-missing to render placeholders instead.")
