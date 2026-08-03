# ssi_figure_v4 derived-refresh handoff manifest

Purpose: give a collaborator on upstream `main` enough files to regenerate
`outputs/fig/ssi_figure_v2/ssi_figure_v4.pdf`, either from the saved solo-machine
caches or by refreshing the derived analysis tables those figure panels read.

This is intentionally an overlay manifest, not a partial git branch. The local
analysis branch adds a much larger tree; this file marks the smaller source and
cache boundary for the `ssi_figure_v4` path.

## Scope

The target is the derived-analysis refresh layer:

- B/C/E/F path-bin tables.
- G alternative dose-axis tables.
- I edge-coherence motion summaries.
- J behavior/model bridge summaries.
- K patch-radius summary.
- A/D schematic and coherence-gallery support caches.

This does not try to package every upstream raw rerun, every diagnostic figure,
or the full local active-sensing/fixation-statistics sandbox.

## Baseline assumptions

- Ryan starts from upstream `main`.
- The working checkout is on the same solo machine, so large existing `outputs/`
  caches can be read in place or copied/symlinked.
- Repo-relative paths below are relative to the repository root.
- Local source used to build this manifest was inspected from local `main`
  around commit `544a474`; upstream comparison point was `upstream/main` around
  `97ce32f`.

## Python dependencies

Do not copy the local `pyproject.toml` wholesale unless you also want its local
DataYates/DataRowley path edits. For this figure overlay, patch upstream deps
minimally:

```toml
"pypdf>=6.14.2",
"pymupdf>=1.28.0",
```

`pypdf` is required by `compose_ssi_figure_v4.py` and panel A compositing.
`pymupdf` is required only by optional PDF extraction/layout helpers; if the
cached `panel_a_network_icon.pdf` and layout override JSONs are shipped, Ryan
should not need to run those helpers for a normal figure rebuild.

## Required source overlay

These are the source files from the recursive local import closure of the v4
composer plus the refresh scripts for the derived tables. Copy them over the
upstream checkout, preserving paths.

### Figure compositor and panels

```text
declan/fig/ssi_figure_v2/compose_ssi_figure_v4.py
declan/fig/ssi_figure_v2/panels/__init__.py
declan/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence.py
declan/fig/ssi_figure_v2/panels/build_coherence_gallery_cache.py
declan/fig/ssi_figure_v2/panels/extract_panel_a_network_icon.py
declan/fig/ssi_figure_v2/panels/panel_a_motion_schematic.py
declan/fig/ssi_figure_v2/panels/panel_bcef_path_bins.py
declan/fig/ssi_figure_v2/panels/panel_d_contour_relative_stimulus.py
declan/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic.py
declan/fig/ssi_figure_v2/panels/panel_g_matched_bins_bracket.py
declan/fig/ssi_figure_v2/panels/panel_g_option_sheet.py
declan/fig/ssi_figure_v2/panels/panel_g_rms_excursion.py
declan/fig/ssi_figure_v2/panels/panel_h_unwrapped_edge_coherence.py
declan/fig/ssi_figure_v2/panels/panel_header.py
declan/fig/ssi_figure_v2/panels/panel_j_match_advantage.py
declan/fig/ssi_figure_v2/panels/panel_k_patch_radius_alignment_slope.py
declan/fig/ssi_figure_v2/panels/reference_layout_v3.py
```

### Behavior/model bridge

```text
declan/fig/ssi_figure_v2/behavior_model_bridge/plot_bridge_explainer_figure.py
declan/fig/ssi_figure_v2/behavior_model_bridge/run_behavior_model_bridge.py
declan/fig/ssi_figure_v2/behavior_model_bridge/run_random_rotation_match_null.py
declan/fig/ssi_figure_v2/behavior_model_bridge/run_random_rotation_prediction_by_coherence.py
```

### Panel A/D schematic source

```text
declan/fig_ssi/make_ssi_contour_schematic.py
```

### Active-sensing derived-analysis helpers

```text
declan/active_sensing_movie_information/make_backimage_component_2d_surface_diagnostic.py
declan/active_sensing_movie_information/make_backimage_component_path_baseline_decomposition_surface.py
declan/active_sensing_movie_information/make_backimage_panel_b_orientation_match_15deg.py
declan/active_sensing_movie_information/make_backimage_panel_b_orientation_match_15deg_sf05.py
declan/active_sensing_movie_information/make_backimage_panel_c_across_along_tail_contrast.py
declan/active_sensing_movie_information/make_backimage_panel_c_sf05_cell_baseline_errorbars.py
declan/active_sensing_movie_information/make_backimage_reordered_geometry_story_figure.py
declan/active_sensing_movie_information/make_backimage_reordered_geometry_story_figure_cell_baseline_sf075_coh020_cde8bins.py
declan/active_sensing_movie_information/plot_backimage_real_trace_unit_first_and_population_schematics.py
```

### Fixation-statistics derived-analysis helpers

```text
declan/fixation_statistics_by_stimulus/__init__.py
declan/fixation_statistics_by_stimulus/extraction.py
declan/fixation_statistics_by_stimulus/features.py
declan/fixation_statistics_by_stimulus/image_features.py
declan/fixation_statistics_by_stimulus/io_utils.py
declan/fixation_statistics_by_stimulus/plot_backimage_contour_motion_components.py
declan/fixation_statistics_by_stimulus/run_backimage_image_structure_analysis.py
declan/fixation_statistics_by_stimulus/run_backimage_twin_drift_geometry.py
declan/fixation_statistics_by_stimulus/summarize_backimage_patch_radius_sensitivity.py
```

### Inherited local dependency to verify

```text
jake/twininfo/eye_controls.py
```

This is in the local import closure through the fixation-statistics helpers.
If upstream `main` already has it, do not send it. If Ryan's checkout lacks it,
include this one file as part of the overlay.

## Optional provenance/reference files

These are useful for auditing, layout regeneration, or old v2/v3 comparisons,
but are not required for a cache-first v4 rebuild once the required source and
support caches are present.

```text
declan/fig/ssi_figure_v2/README.md
declan/fig/ssi_figure_v2/ssi_figure_v2_3.pdf
declan/fig/ssi_figure_v2/generate_ssi_figure_v2.py
declan/fig/ssi_figure_v2/compose_ssi_figure_v3.py
declan/fig/ssi_figure_v2/panels/extract_reference_layout.py
declan/fig/ssi_figure_v2/panels/page_layout_boxes.py
declan/fig/ssi_figure_v2/panels/panel_a_layout_boxes.py
declan/fig/ssi_figure_v2/panels/panel_d_coherence_gallery.py
declan/fig/ssi_figure_v2/panels/panel_d_layout_boxes.py
declan/fig/ssi_figure_v2/panels/panel_g_layout_boxes.py
declan/fig/ssi_figure_v2/panels/panel_g_local_contour_detail.py
declan/fig/ssi_figure_v2/panels/panel_g_relation_sweep_matched_bins.py
declan/fig/ssi_figure_v2/panels/panel_i_edge_alignment.py
declan/fig/ssi_figure_v2/panels/svg_box_utils.py
declan/fig/ssi_figure_v2/behavior_model_bridge/README.md
declan/fig/ssi_figure_v2/behavior_model_bridge/compile_behavior_model_bridge_diagnostic_compendium.py
declan/fig/ssi_figure_v2/behavior_model_bridge/plot_random_rotation_match_null_single_summary.py
```

## Cache-first artifacts

If the goal is to reproduce the existing v4 figure with minimal compute, send
or point Ryan at these small derived artifacts. These are the CSV/NPZ/PDF/JSON
inputs the final figure panels read directly or nearly directly.

### Figure-value tables

```text
outputs/fig/ssi_figure_v2/panels/panel_bcef_path_bins_values.csv
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_values.csv
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_last_bin_contrasts.csv
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_trace_bank_reference.csv
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_populations.csv
outputs/fig/ssi_figure_v2/panels/panel_h_unwrapped_edge_coherence_values.csv
outputs/fig/ssi_figure_v2/panels/panel_h_unwrapped_edge_coherence_random_orientation_reference.csv
outputs/fig/ssi_figure_v2/panels_v3/panel_k_patch_radius_alignment_slope_values.csv
```

### Behavior/model bridge tables

```text
outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence_windows.csv
outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence_summary.csv
outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence_alignment_summary.csv
outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence_directional_path_random_orientation_reference.csv
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_coherence_contrasts.csv
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_prediction_summary.csv
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_random_rotation_match_null_rotation_values.csv
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_random_rotation_match_null_session_values.csv
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_random_rotation_match_null_summary.csv
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_random_rotation_prediction_by_coherence_session_values.csv
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_random_rotation_prediction_by_coherence_summary.csv
```

### Schematic and layout support caches

```text
outputs/fig_ssi/trace_provenance/schematic_crop_real_backimage_trace_center40.csv
outputs/fig_ssi/trace_provenance/schematic_crop_real_backimage_trace_full128.csv
outputs/fig_ssi/rr100_schematic_endpoint_final_maps/cache/schematic_rr100_final_maps.npz
outputs/fig_ssi/rr100_schematic_endpoint_final_maps/schematic_rr100_final_map_unit_metrics.csv
outputs/fig/ssi_figure_v2/panels/cache/coherence_gallery.npz
outputs/fig/ssi_figure_v2/panels/cache/panel_a_layout_overrides.json
outputs/fig/ssi_figure_v2/panels/cache/panel_d_layout_overrides.json
outputs/fig/ssi_figure_v2/panels/cache/panel_a_network_icon.pdf
```

With the required source overlay plus these cache-first artifacts, Ryan should
be able to run:

```bash
uv run python declan/fig/ssi_figure_v2/compose_ssi_figure_v4.py
```

Expected generated outputs, not bundle inputs:

- `outputs/fig/ssi_figure_v2/ssi_figure_v4.pdf`
- `outputs/fig/ssi_figure_v2/ssi_figure_v4_provenance.json`
- `outputs/fig/ssi_figure_v2/panels_v3/`

## Large lower-level caches for derived refresh

If Ryan wants to regenerate the figure-value tables instead of using the small
cache-first artifacts, these larger outputs need to exist at the same
repo-relative paths, or be symlinked there.

```text
outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/
outputs/active_sensing_movie_information/backimage_rr100_instantaneous_unit_maps_latest_v1/
outputs/active_sensing_movie_information/backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered_yfix/
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered_yfix_slope_v1/
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/window_features.csv
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_patch_radius_0p25_v1/
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_patch_radius_0p5_v1/
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_patch_radius_sensitivity_v1/
```

Approximate local sizes of the biggest reusable cache roots:

```text
846M  outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1
597M  outputs/active_sensing_movie_information/backimage_rr100_instantaneous_unit_maps_latest_v1
74M   outputs/active_sensing_movie_information/backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1
72M   outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1
29M   outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_patch_radius_sensitivity_v1
52M   outputs/fig/ssi_figure_v2
29M   outputs/fig_ssi
```

## Derived refresh run order

Run from repository root. The first command rebuilds the B/C/E/F source table;
the remaining commands refresh G, I/J support, K, optional D cache, then compose
the final figure.

```bash
uv run python declan/active_sensing_movie_information/make_backimage_panel_b_orientation_match_15deg_sf05.py
uv run python declan/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic.py
uv run python declan/fixation_statistics_by_stimulus/plot_backimage_contour_motion_components.py --no-recompute-traces
uv run python declan/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence.py
uv run python declan/fig/ssi_figure_v2/behavior_model_bridge/run_behavior_model_bridge.py
uv run python declan/fig/ssi_figure_v2/behavior_model_bridge/run_random_rotation_match_null.py
uv run python declan/fig/ssi_figure_v2/behavior_model_bridge/run_random_rotation_prediction_by_coherence.py
uv run python declan/fixation_statistics_by_stimulus/summarize_backimage_patch_radius_sensitivity.py
uv run python declan/fig/ssi_figure_v2/panels/build_coherence_gallery_cache.py
uv run python declan/fig/ssi_figure_v2/compose_ssi_figure_v4.py
```

Notes:

- `build_coherence_gallery_cache.py` is optional if
  `outputs/fig/ssi_figure_v2/panels/cache/coherence_gallery.npz` is present.
- `plot_backimage_contour_motion_components.py --no-recompute-traces` assumes
  the input windows already contain trace columns. Drop that flag if Ryan needs
  to recompute traces from the underlying data.
- The bridge random-rotation scripts are bootstrap/null steps; they are the
  slower part of the refresh path but still downstream of the saved behavior
  and model tables.
- `summarize_backimage_patch_radius_sensitivity.py` expects the 0.25, 0.5, and
  1.0 deg radius outputs listed in the large-cache section, and uses the
  existing `backimage_patch_radius_sensitivity_v1` cache when available.

## Packaging recommendation

For minimal implementation overhead, send two bundles:

1. Code overlay: the required source overlay above, plus the optional
   provenance/reference files only if Ryan wants audit/layout tooling.
2. Cache overlay: the cache-first artifacts above. Because Ryan is on solo, this
   can be a manifest of paths plus symlinks instead of a large tarball.

Keep the large lower-level caches out of the handoff package unless Ryan is not
working from the shared solo outputs. They are useful for refresh, but they are
not part of a small code-file transfer.
