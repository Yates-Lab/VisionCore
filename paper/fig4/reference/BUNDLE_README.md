# SSI figure v2

Draft figure-generation folder for a multipanel SSI figure based on
`SSI_figure_v2_rkr (1).pdf`.

The generator is currently a composition scaffold with selected existing
analysis panels pulled in. It combines:

- schematic FEM/stabilized movie blocks, response maps, and stimulus/crop assets
  from `declan/fig_ssi/make_ssi_contour_schematic.py`,
- cell-baselined BackImage trajectory-path panels for B/C/E/F from
  `declan/active_sensing_movie_information/make_backimage_panel_b_orientation_match_15deg_sf05.py`
  (E/F are drawn as insets inside D's own axes, not a separate gridspec cell
  -- see `EF_INSET_*` constants and `draw_panel_a`),
- a reference crop plus the zoomed local-contour aperture for G (see
  `draw_contour_components_panel`; G used to hold D's former unfinished
  right half, a reserved contour-normal/parallel decomposition placeholder
  ("Contour-carried signal") -- that's gone, replaced by moving D's third
  cascaded image (the zoomed trace-aperture detail) here along with a
  reference copy of D's 151x151 crop, since the zoom was competing with D's
  E/F insets for width. The crop/zoom overlap is kept proportionally
  identical to D's original, expressed as a fraction of the crop's own size
  since this axes has a different physical aspect than D's. Unit-tuning
  alignment (B/C/E/F/H's "aligned" population) and eye-trace alignment (D's
  aperture, H's across/along axis, I, J) remain two unrelated senses of
  "aligned with the contour". G's lower half also decomposes this same
  example trace into across-/along-contour components -- the same split
  H's dose curve reports as separate lines -- via
  `draw_rms_excursion_explainer`, using this one trace's real geometry as
  an illustration, not H's population statistic),
- a local-edge-coherence example gallery in D's lower-left (see
  `panels/panel_d_coherence_gallery.py` and
  `panels/build_coherence_gallery_cache.py`; replaces D's former
  short/long-path trace legend, since the real traces moved to G where
  there's room to show them at a legible scale. One real BackImage crop is
  cached per `COHERENCE_ORDER` bin, picked closest to that bin's midpoint
  coherence value, and colored with the same low->high sequential palette
  as I's coherence-bin legend (`panel_h_unwrapped_edge_coherence.COLORS`).
  Rebuild the cache with `uv run python
  declan/fig/ssi_figure_v2/panels/build_coherence_gallery_cache.py` (slow,
  ~20s per example -- reads real stimulus frames via DataYatesV1 -- so it's
  cached rather than recomputed on every figure render)),
- the aligned high-SF RMS-excursion errorbar/bracket panel for H (see
  `panels/panel_g_rms_excursion.py`; this replaced the original
  unsigned-component-path version in `panels/panel_g_matched_bins_bracket.py`
  once the behavior-model bridge showed path has no support from real
  behavior while RMS excursion does -- see
  `behavior_model_bridge/README.md` and `panels/panel_g_option_sheet.py`),
- a reduced-bin unwrapped contour-relative position-spread panel for I, and
- the coherence-resolved trace-contour match-advantage panel for J (see
  `panels/panel_j_match_advantage.py`; this replaced the descriptive
  Figure-4E-style edge-coherence alignment panel in
  `panels/panel_i_edge_alignment.py`, kept unwired for reference, once the
  random-rotation null showed the coherence-dependent correlation is also
  model-beneficial relative to chance -- see `behavior_model_bridge/README.md`).

Panel letters G/H/I/J were G/H/I before the reserved decomposition panel was
inserted between F and the RMS-excursion panel; the underlying scripts/files
still say "g"/"h"/"i" in their names since renaming them added no value, only
the displayed `label` default in each `draw_panel()` changed.

## Typography

Use these sizes as the default style guide for manual refinement on an 8.5 x
11 inch page:

- Main figure title: 13.5 pt, bold.
- Figure subtitle/provenance line: 8.5 pt, gray.
- Large section headers for A and D: 14 pt panel letter, 12 pt bold section
  title.
- Quantitative panel titles for B/C/E/F/H/I/J: 8.5-9 pt, bold, colored only
  when the panel encodes low-SF blue or high-SF orange. G (image assets, not
  a quantitative result) uses the same size but is never colored, like H/I/J
  when they mix populations.
- Axis labels: 7-8 pt. Keep compact path-bin panels near 7 pt; use 8 pt for
  roomier standalone axes.
- Tick labels: 6.8-7 pt.
- Legends and unit/pair support notes: 5.7-6.1 pt.
- Schematic annotations: 5.2-7.2 pt, with the smallest text reserved for
  diagram labels rather than claims/results.
- Avoid adding threshold definitions to the figure itself; keep those in the
  methods/provenance JSON unless a panel absolutely needs them.

The live panels read precomputed values from:

```text
outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/phase1_phase2_conditioning_v1/plot_collections/
```

Panel-specific scripts live in `panels/` and can be run independently:

```bash
uv run python declan/fig/ssi_figure_v2/panels/panel_d_contour_relative_stimulus.py
uv run python declan/fig/ssi_figure_v2/panels/panel_bcef_path_bins.py
uv run python declan/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic.py  # regenerates the values/contrasts CSVs below
uv run python declan/fig/ssi_figure_v2/panels/panel_g_rms_excursion.py  # current Panel H
uv run python declan/fig/ssi_figure_v2/panels/panel_g_matched_bins_bracket.py  # original path-based version, no longer wired in
uv run python declan/fig/ssi_figure_v2/panels/panel_g_option_sheet.py  # the four-axis comparison that motivated the switch
uv run python declan/fig/ssi_figure_v2/panels/panel_h_unwrapped_edge_coherence.py  # current Panel I
uv run python declan/fig/ssi_figure_v2/panels/panel_j_match_advantage.py  # current Panel J
uv run python declan/fig/ssi_figure_v2/panels/panel_i_edge_alignment.py  # original descriptive version, no longer wired in
```

Their data sources are:

```text
outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/phase1_phase2_conditioning_v1/plot_collections/backimage_real_trace_panel_b_cell_baseline_sf05_coh020_match15_{values,selection_summary,summary}.*
outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/phase1_phase2_conditioning_v1/plot_collections/backimage_real_trace_panel_c_aligned_sf_ge_0p5_match15_matched_bins_bracket_{values,last_bin_contrast,summary}.*  (original path-based version of current H, still used by panel_g_matched_bins_bracket.py)
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_{values,last_bin_contrasts,trace_bank_reference,populations}.csv  (current H, RMS excursion)
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_random_rotation_match_null_summary.csv  (behavior bridge evidence used to choose the RMS axis)
outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_random_rotation_prediction_by_coherence_summary.csv  (current J, match advantage by coherence)
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_position_spread_unwrapped_profiles_by_edge_coherence.csv
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv  (original descriptive I, still used by panel_i_edge_alignment.py)
```

Panel D draws the local contour-axis/coherence aperture from
`stimulus_row.image_patch_radius_px`. In the reviewed BackImage window table
this radius is 38 px; with `MODEL_PPD = 37.50476617` this is 1.01 deg, so the
schematic labels it as a 1 deg radius local window. The left side then adds a
second-stage center crop around fixation so the FEM traces are legible without
changing the analysis aperture. That trace zoom uses `+/-0.25 deg` as a minimum
and expands just enough to contain the selected trace pair when needed.

B/C/E/F and H intentionally share the same methods-facing selection definitions:
`sf_split_metric < 0.5` for low-SF units, `sf_split_metric >= 0.5` for
high-SF/aligned high-SF units, image contour coherence `>= 0.20`, orientation
selectivity `>= 0.05` for aligned unit-image matching, and unit-contour
orientation difference `<= 15 deg` for aligned panels. These details are not
printed on the figure; the generator writes them to:

```text
outputs/fig/ssi_figure_v2/ssi_figure_v2_methods_provenance.json
```

## Verbose guide: what each analysis is doing

This figure now combines two related but non-identical questions. The SSI
panels ask what different kinds of FEM trajectories do to neural information in
the BackImage model/movie analysis. The behavioral contour-relative panels ask
how the animal's actual fixation drift geometry changes when the local image
contains a coherent oriented contour. These are linked by the same visual axes,
but they do not always use the same movement metric.

The most important distinction is between accumulated path and position spread.
Accumulated path is a distance-walked metric. It sums sample-to-sample motion
over the trace, so a back-and-forth trace can have a large path length even if
it stays in a narrow region of space. Position spread is a cloud-width metric.
It asks how far the set of eye positions extends around its own center. A trace
can therefore have high accumulated path but low spread when the motion reverses
often, and it can have lower accumulated path but larger spread if it moves
more persistently away from the cloud center.

For B/C/E/F, the x-axis is total 2D FEM path length. For each trace, we sum the
Euclidean step sizes between consecutive eye samples:

```text
total path length = sum_t sqrt(dx_t^2 + dy_t^2)
```

The trace is then assigned to a path-length bin, and the y-axis is the
cell-baselined change in single-spike information, plotted as percent change
relative to a cell-matched stabilized baseline with matching image composition.
These panels do not ask whether the trace is moving along or across a contour.
They ask whether more total FEM path helps or hurts SSI for different unit and
image selections.

Panel B selects low spatial-frequency units on strong-contour images without an
orientation-alignment requirement. In the current values, the drift-only bins
rise from about 16.0% to 35.9% SSI change as path length increases. The
interpretation is that low-SF responses benefit from larger FEM paths, probably
because broader retinal motion samples image structure in a way that improves
the low-SF signal.

Panel C uses high spatial-frequency units on the same strong-contour image
selection, again without the unit-contour alignment requirement. Here the
drift-only bins are nearly flat to mildly decreasing, about 14.6% down to
12.4%. The interpretation is that high-SF information is less helped by extra
motion and may begin to be degraded by longer paths, but the effect is not as
strong as in the aligned high-SF case.

Panel E selects low-SF units whose preferred orientation is aligned with the
local contour. The drift-only bins rise from about 7.5% to 19.0%. This is still
a low-SF benefit from path length, but with a lower starting point and smaller
increase than Panel B. The aligned subset is a stricter unit-image population,
so it should be interpreted as a conditional low-SF result, not as a duplicate
of Panel B.

Panel F selects high-SF units whose preferred orientation is aligned with the
local contour. This is where total path length becomes clearly harmful: the
drift-only bins fall from about 3.6% to -8.2%. This is the first strong sign
that high-SF contour-aligned information is vulnerable to longer FEM paths.
However, Panel F still uses total 2D path length, so it does not yet identify
which contour-relative direction is most damaging.

Panel H decomposes the high-SF aligned case by contour-relative component path.
For each local contour, we define two axial directions: the contour tangent
("along") and the contour normal ("across"). For each eye-movement step, we
project the 2D step onto one of those axes, take the absolute value, and sum
over time:

```text
along component path  = sum_t abs(dot([dx_t, dy_t], tangent_axis))
across component path = sum_t abs(dot([dx_t, dy_t], normal_axis))
```

The absolute value is important. Component path is still an accumulated
distance-walked metric. A trace that repeatedly steps across the contour and
then steps back will accumulate across-component path even if its final
displacement and position spread across the contour are small.

Panel H asks whether SSI loss in high-SF aligned units is worse when accumulated
path is across the contour or along it. The current values show that the
across-contour component is more harmful: the across trace falls from about
+3.4% to -14.7%, while the along trace falls from about +1.9% to -4.9%. The
last-bin across-minus-along contrast is about -9.9 percentage points
(`p = 0.0002`). This supports the model-side claim that high-SF contour-aligned
SSI is especially vulnerable to contour-normal motion.

A model-side alternative-axis diagnostic keeps the same cell-matched baseline,
but replots Panel-G-style marginal curves with different contour-relative dose
axes across five populations: all high-SF units, aligned high-SF units, oblique
high-SF units, orthogonal high-SF units, and all low-SF units.

```text
declan/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic.py
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic.{png,pdf,svg}
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_{high_sf_all,high_sf_aligned,high_sf_oblique,high_sf_orthogonal,low_sf_all}.{png,svg}
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_values.csv
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_last_bin_contrasts.csv
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_populations.csv
outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_trace_bank_reference.csv
```

Each panel includes a gray x-axis band showing the drift-only BackImage
real-trace-bank q25-q75 range, pooled across contour-normal and
contour-parallel projections while ignoring unit tuning and trace-contour
alignment. Using the same tail-enriched binning style, the aligned high-SF
contour-normal penalty persists when the x-axis is unsigned component path,
component RMS excursion, or projected peak-to-peak range. The final-bin
normal-minus-parallel contrasts for aligned high-SF units are about -9.9 pp,
-7.6 pp, and -5.6 pp respectively, all with bootstrap `p < 0.001`. Oblique
high-SF units show a weaker same-sign normal penalty, while orthogonal high-SF
units flip sign, as expected if the vulnerable motion axis follows the unit's
orientation relationship to the contour. The path/range tortuosity proxy has
little aligned high-SF normal-parallel separation. This makes the bridge to I/J
more plausible: the damaging model axis is not only accumulated normal path, but
also large normal excursion/spread. The behavioral claim still has to be phrased
in terms of spread/covariance rather than accumulated path.

Panels I/J and the associated diagnostics switch to the animal's real
contour-relative behavior. Here the main measurement is not accumulated path
length. Panel I is an unwrapped position-spread profile. For each reviewed
BackImage window, the eye positions are centered, projected onto axes at
different angles relative to the local contour, and summarized by RMS:

```text
projected position spread = sqrt(mean((projected_position_t - mean_projected_position)^2))
```

In this convention, 0/180 deg are contour-parallel axes and 90 deg is the
contour-normal axis. As local edge coherence increases, the RMS profile becomes
more elongated along the contour and narrower across it. For example, in the
0.8-1 coherence band, the parallel RMS is about 2.11 arcmin and the normal RMS
is about 1.76 arcmin. That is a statement about the shape of the eye-position
cloud.

Panel J compresses that same geometry into an edge-alignment index. The
underlying idea is covariance alignment: estimate the dominant axis of the
drift-position cloud, compare it to the local contour axis, and use an axial
cosine-style score so that parallel directions count together regardless of
sign. The index increases with coherence, from near 0.06 in the lowest bin to
roughly 0.30 in the 0.7-0.8 bin, with the 0.9-1.0 bin remaining suggestive but
sparser. This says that coherent contours are associated with more
contour-parallel drift-cloud orientation.

The diagnostic plots add the missing bridge between Panel H and Panels I/J.
They compute the same contour-relative components on the real behavioral traces,
but they show both accumulated component path and position spread. The result is
the key interpretive caution: high-coherence windows show reduced orthogonal
spread, but they do not necessarily show reduced orthogonal accumulated
component path. In some bins, the normal component can accumulate as much or
more unsigned path than the parallel component.

This is not contradictory. It means the animal's trace can have similar local
step distance in each projected direction while those steps produce different
cloud shapes. If the normal steps are more reversing or more confined, they add
to accumulated path but do not make the position cloud wide across the contour.
Thus, the behavioral statement should be about reduced contour-normal
excursion/spread and increased contour-parallel covariance alignment, not about
the animal simply walking less distance across the contour.

The combined figure should therefore be read in this order:

1. B/E establish that low-SF SSI benefits from longer FEM path length.
2. C/F show that high-SF SSI is not helped in the same way, and high-SF
   contour-aligned SSI is harmed by long paths.
3. H shows that, within high-SF aligned units, accumulated contour-normal path
   is more damaging than accumulated contour-parallel path.
4. I/J show that real behavior near coherent contours creates a drift cloud
   that is contour-parallel and comparatively narrow in the contour-normal
   direction.
5. The diagnostics show that I/J should not be rewritten as an accumulated-path
   claim. The axis correspondence is real, but the movement metric differs.

The conservative biological interpretation is:

```text
Low-SF units appear to benefit from larger FEM paths.
High-SF contour-aligned units are vulnerable to motion across the contour.
In real fixations, coherent contours are associated with drift clouds that are
elongated along the contour and confined across it.
This may help preserve high-SF contour information by limiting cross-contour
excursion/spread, even when accumulated cross-contour step distance is not
strongly reduced.
```

The claim to avoid is:

```text
Animals move farther along contours and less across contours.
```

That wording collapses spread/covariance and accumulated path length into one
idea. The data we have support a subtler claim: the real drift cloud becomes
more contour-parallel, while the accumulated-path diagnostics remain a separate
control showing how much back-and-forth motion occurs along each projected
axis.

## Interpretation note: component path is not behavioral alignment

Panel H uses unsigned component path length as the model dose axis:
`sum(abs(projected sample-to-sample eye displacement))` along the local contour
tangent or normal axis. This is the right quantity for the model trace bank in
that panel, but it is not the same thing as saying the animal's eye movement
trajectory is contour-parallel in the behavioral data.

The behavioral alignment result in I/J is a covariance/position-spread result:
as local contour coherence increases, the drift cloud is more elongated along
the contour tangent than the contour normal, and `drift_edge_cos2` increases.
The panel wording should therefore emphasize position spread, drift-cloud
orientation, and spread-axis alignment, not edge-following path length.
A separate diagnostic makes this distinction explicit:

```text
declan/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence.py
outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence.{png,pdf,svg}
outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence_alignment_summary.csv
outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence_directional_path_profile.{png,pdf,svg}
outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence_spread_vs_directional_path_profile.{png,pdf,svg}
```

Run it from the repository root with:

```bash
uv run python declan/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence.py
```

On the same reviewed BackImage windows, contour-parallel RMS exceeds
contour-normal RMS at high coherence, but unsigned accumulated component path
can be slightly larger in the normal direction. This is not evidence for a
tangent/normal label swap. It means that path length and excursion/covariance
answer different questions: a trace can have a contour-parallel position cloud
while accumulating more normal-direction step length through small back-and-forth
jitter. Figure text should therefore avoid saying that behavior increases
"along-contour path length"; the defensible statement is that behavior increases
contour-parallel position spread/alignment, while Panel H tests how SSI changes
with unsigned component path length in the model trace bank.

The directional-path diagnostics use the same unwrapped convention as Panel I:
0/180 deg are contour-parallel axes and 90 deg is the contour-normal axis. They
show why path length should stay a diagnostic/control rather than replace I/J:
the RMS profile is lowest at the normal axis in high-coherence windows, whereas
the unsigned component-path profile can be highest near that same normal axis.
For example, in the 0.8-1 coherence band, normal-minus-parallel RMS is about
-0.35 arcmin, while normal-minus-parallel unsigned component path is about
+1.46 arcmin per 0.325 s.

A closer BackImage follow-up decomposes that path effect using the same 0.1-wide
coherence bins and unwrapped edge-relative format as the position-spread Panel B
follow-ups:

```text
declan/fixation_statistics_by_stimulus/generate_backimage_contour_signed_path_followups.py
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_signed_path_components_unwrapped_stack_by_edge_coherence.{png,pdf}
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_signed_path_component_progression_by_edge_coherence.{png,pdf}
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_signed_path_component_progression_by_edge_coherence.csv
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_unsigned_path_vs_position_spread_rms_by_edge_coherence.{png,pdf}
```

It writes three quantities: signed net component, absolute net component, and
unsigned accumulated component path, each scaled to the 0.325 s equivalent
window used by the path-length diagnostic. In the high-coherence bins, the
absolute net component is larger along the contour-parallel axis, while the
unsigned path can be larger near the orthogonal axis. This supports the
interpretation that the apparent orthogonal path excess is largely
back-and-forth step accumulation rather than a sustained orthogonal excursion.
Because local contour orientation is axial, the signed net component is only a
cancellation/polarity diagnostic; its sign should not be read as a stable
biological direction.

The path/spread interpretation can be audited with a broader diagnostic pack.
These plots use the same fixed 0.1-wide coherence bins and bin centers as Panel
I:

```text
declan/fixation_statistics_by_stimulus/diagnose_backimage_contour_path_spread.py
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_scale_diagnostics_by_edge_coherence.{png,pdf}
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_component_ratio_diagnostics_by_edge_coherence.{png,pdf}
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_step_reversal_diagnostics_by_edge_coherence.{png,pdf}
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_msd_growth_by_edge_coherence.{png,pdf}
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_final_bin_path_length_influence.{png,pdf}
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_final_bin_vs_overall_path_length.{png,pdf}
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_scale_diagnostics_by_orientation_energy.{png,pdf}
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_scale_diagnostics_by_coherent_orientation_energy.{png,pdf}
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_scale_diagnostics_by_coherence_x_orientation_energy.{png,pdf}
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_scale_axis_comparison_projection_overlay.{png,pdf}
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_scale_axis_comparison_2d_trace.{png,pdf}
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_step_diagnostics_summary.csv
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_final_bin_path_length_*.csv
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_final_bin_vs_overall_path_length_summary.csv
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_path_spread_scale_diagnostics_by_*orientation_energy*_summary.csv
```

These diagnostics show the scale separation directly: the parallel/orthogonal
RMS ratio increases with coherence, while the parallel/orthogonal unsigned-path
ratio stays near one or slightly below one. In other words, the orthogonal
component does not simply lose projected distance; rather, similar local
projected distance appears to produce less orthogonal spread, consistent with
stronger back-and-forth cancellation or confinement across the contour.
The scale diagnostic also includes an unprojected 2D trace row: RMS is the
Euclidean radius of the centered eye-position cloud, cumulative path length is
the Euclidean accumulated path over the same trace samples, and net fraction is
the Euclidean start-to-end displacement divided by that cumulative path length.
The energy-axis follow-ups use quantile bins because local gradient energy is
heavy-tailed. `orientation_energy` is the raw Sobel gradient energy; it measures
local visual signal strength but can be high for multi-orientation texture.
`coherent_orientation_energy` is `orientation_energy * orientation_coherence`,
the anisotropic part of the local structure-tensor energy. The
coherence-by-energy split suppresses cells with fewer than 10 sessions or 30
windows so the sparse 0.9-1.0 coherence band does not dominate the plot.

Run from the repository root:

```bash
uv run python declan/fig/ssi_figure_v2/generate_ssi_figure_v2.py
```

Outputs are written to:

```text
outputs/fig/ssi_figure_v2/ssi_figure_v2.{png,pdf,svg}
outputs/fig/ssi_figure_v2/ssi_figure_v2_panel_boxes.svg
outputs/fig/ssi_figure_v2/panels/panel_d_contour_relative_stimulus*.{png,pdf,svg}
```

`ssi_figure_v2_panel_boxes.svg` is an editable empty-box composition template
with one SVG group per panel (`panel-A` through `panel-I`). It has two box
layers:

- blue `panel-axes-boxes`: raw Matplotlib axes positions.
- orange `panel-axis-text-boxes`: axes plus tick labels and axis labels, but
  excluding panel headings/titles.

Move or resize those boxes during manual layout refinement, then pass the edited
SVG back as a guide.
