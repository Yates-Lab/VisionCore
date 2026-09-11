# Completed rank-one manuscript migration — 2026-09-11 UTC

The active manuscript now uses the completed rank-one model without the separate
first-stage phase readout. Figures 3 and 4, their model-dependent statistics,
feature-map illustrations, and Methods use checkpoint
`e70f287462155607a96c7d23d9d56e347111319aa48cafd3f54fe88dd4484cb9`.
The selected analysis is `outputs/no_phase_readout_comparison_20260910/rank1`;
`analysis/selected_model_bundle.json` pins its completed results.

Figure 3 was regenerated and checked against an independent held-out prediction
audit. Figure 4 includes new tuning for 725 exact checkpoint units (145 passing
the strict illustrative-example gates), the complete 40-image by 200-history
response replay, matched stabilized baselines, both direct spectral batches,
passband comparison, and the 100-pair cumulative-stage analysis. C/F compare
47 drift windows with 18 verified subdegree-microsaccade windows after continuous
6-ms Gaussian filtering and equal-animal, equal-dynamic-mass normalization.
The main numerical conclusions remain supported. Results and captions now
report the new model's quantities. The path-length reference is explicitly the
sixth of eight equal-count bins; stage claims describe cumulative readouts.

The replay audit exposed a units error: native model outputs are expected counts
per 1/240-s bin. The source reducer now converts their mean to spikes/s and sums
bin counts without an extra duration factor. Saved rates and accumulated counts
were repaired by the verified factor of 240; SSI and spectral arrays were
preserved. All dependent reductions and the direct cumulative-stage replay were
rerun. The stage output matches the ordinary model, and its rates, counts and
SSI match the repaired spectral cache within the recorded tolerances. Repair
backups and independent checks are in the bundle's `manuscript_migration/` folder;
its initial spectral-only correction is explicitly superseded by the source-unit
repair. Training and held-out Figure 3 predictions were unaffected.

Validation: 20 targeted regression tests passed; complete response/spectral
integrity and production-release checks passed; `make -C manuscript audit`
passes after the final figure layout review. All 95 generated statistics are
verified. The compiled PDF has 40 pages and its smallest measured figure font
is 6.087 pt. Figures 3/4 (pages 8/12) and their captions (pages 9/13) were visually
inspected. Shared/wrapped axis labels prevent crowding in Figure 4B/G/H.
The interim spectrum selection and temporary model-update notes were removed.
Figures 1/2 and both supplements retain their empirical analyses.

`analysis/model_migration_audit.json` records completion, the selected model,
review and validation hashes, and the final PDF hash. Build logs are
`build/rank1_installation.log`, `build/rank1_final_layout.log`, and
`build/rank1_final_audit.log`. The build remains `make -C manuscript`.

Everything below is historical revision history. Statements about pending runs,
architecture previews, retained old-model results, and previous PDF hashes apply
only to those earlier revisions.

# Figure 3 input simplification and FixRSVP examples — 2026-09-11 UTC

Each shared input now uses a single arrow: neutral for the measured retinal
history and green for the extraretinal history. The blue Full label, red
zero-input intervention, and purple stabilized alternative retain the
hypothesis-testing cues without duplicate arrows or a repeated Stabilized
label on the behavior route.

Panel B uses the paired cached FixRSVP illustrations (60 frames, 35 × 35
pixels), and its stage textures are recomputed from the displayed measured
history with the same checkpoint. The stabilized history preserves image
flashes; the caption and provenance identify its illustrative trial-medoid
anchor and shared current image, distinct from the session-global anchor in
the quantified ablation. Eye traces now use the separate cached FixRSVP
example. Panel A's training illustration remains unchanged.

The input arrays exactly match the cached FixRSVP pair; all feature maps are
finite. Figure 3C--E summary dictionaries and all manuscript statistical
macros are unchanged. The render and manuscript audit pass, including font
sizes, text/image intersections, and installed PDF identity. Source and
compiled pages were visually reviewed. Logs: `build/figure3_fixrsvp_render.log`
and `build/figure3_fixrsvp_audit.log`. Earlier revisions follow below.

# Figure 3 hypothesis-testing schematic — 2026-09-11 UTC

Figure 3B retains the illustrated architecture and restores the three input
conditions in the exact blue/red/purple colors used by C--E. The measured
retinal history feeds Full and Retinal only; the purple fixed-view cube
replaces that history for Stabilized. The separate extraretinal input is
retained for Full and Stabilized, or zeroed before the MLP for Retinal only.
The diagram states that the fitted weights are shared across conditions.
The caption describes these interventions and distinguishes the illustrative
fixed image from the session-global gaze anchor used in the reported ablation.

Model weights, feature-map textures, example prediction values, C--E result
dictionaries, and numerical manuscript macros are unchanged. The existing
no-phase architecture preview and retained-result status still apply.
Separate raster planes remain separate in the vector PDF so that the label
overlap audit can measure the clear space between the two input cubes.
The refreshed figure provenance records the conditions and their colors.

Validation is recorded in `build/figure3_hypothesis_audit.log` and
`analysis/figure_audit.json`, including figure fonts, label intersections,
installed PDF identity, numerical equality, and final manuscript PDF hash.

# Contour spectrum update — 2026-09-11 UTC

Figure 4C/F in the main manuscript now use the approved filled-contour
spectra: 47 drift windows and 18 verified microsaccade-containing windows
with amplitudes below 1 degree, after continuous Gaussian filtering
(6-ms standard deviation). The caption reports these counts and describes
normalization and animal weighting. Results and Methods describe the
revised filtering and event selection.

The motion-response rerun is still in progress. Figure 4A/B/G/H retain the
previous completed response analysis; D/E retain the existing model tuning.
The text, caption, and Methods explicitly identify this temporary distinction.
All 81 previous statistical macros are unchanged; two new macros provide
the audited C/F group counts. Model-dependent results still use the preceding
checkpoint, pending the separate rank-one/rank-two comparison.

`analysis/figure4_spectrum_update.json` pins the spectral input archive,
summary, approved figure, and preview audit. Statistics export, rendering,
and the PDF audit respect that selection. A later completed bundle
supersedes the interim selection and clears its conditional draft notes.
The installed figure matches the approved preview byte-for-byte. Figure 4B–H
summary dictionaries match their respective selected sources exactly.

Validation: `make -C manuscript audit` passes; all six figure pages meet the
6-pt minimum after LaTeX scaling. The compiled Figure 4 and its caption were
visually inspected. The caption now follows the figure instead of floating
to the end of the manuscript. `analysis/figure_audit.json` records the final
PDF hash, page count, fonts, and source identities; the build log is
`build/contour_update_audit.log`.

The remaining sections below document earlier manuscript revisions and
posting checks; their older filtering descriptions and numerical tables are
historical snapshots, not a declaration that the current response rerun has
finished.

# Posting preparation audit — 2026-09-10

Figure 4A layout refinement: a compact vector feature-plane stack now represents
the shared predictive model. Arrows connect both retinal histories to the model,
their activation maps, and the difference map. Rate/SSI labels sit below the
maps; the caption defines their normalization and subtraction. The exemplar
arrays and all numerical results are unchanged.

Figure 3 layout refinement: removed the panel-B sentence heading, reduced the
canvas from 10.6 to 9.6 inches high, aligned column labels above the feature
stacks, and enlarged the C--D gutter. C and D now show significance stars only;
paired effects, confidence intervals, tests, and the star key are in the
caption. Three existing bootstrap p-values are newly exported as macros
(81 total); the previous 78 values and all panel statistics are unchanged.
The figure audit now checks text/image intersections in B as well as text
collisions, fonts, and numerical identity.

## Historical architecture preview with retained results (superseded)

At the user's request, Figure 3B now omits the additional first-stage readout
and its yellow path. Its caption and the architecture/training methods describe
the new rank-one/rank-two ordinary readout comparison. The methods explicitly
locate proximal L1 on the readout's feature/spatial weights during training
phases 2 and 3, and give the Gaussian-envelope and normalization details.

All statistics and result panels remain from the preceding checkpoint listed
below. The feature-map textures and example prediction retain that same source;
they are not outputs of the retraining. Caption/Methods draft notes and separate
source/display architecture records in `analysis/figure3_schematic.json` make
the temporary distinction explicit. Figure 4 retains the preceding model's
results, with its panel-A presentation refined as described above. The final manuscript
must adopt a selected new checkpoint and its completed analyses together.

The completed checks below describe the preceding posting audit; the present
figure audit additionally verifies the explicit architecture preview and
unchanged numerical results.

The four requested manuscript preparation tasks are complete. The active
draft builds to `build/main.pdf`; reproducible commands are in [README.md](README.md).

## Completed checks

- [x] Replace the ConvGRU description with the executed architecture and
  training curriculum, including the native 240-Hz / analysis 120-Hz distinction.
- [x] Audit active results, captions, methods, and interpretation against
  implementation, executed configurations, and completed analysis artifacts.
- [x] Regenerate Figure 4A from the selected checkpoint with central image
  detail, disclose illustrative selection, and preserve population results.
- [x] Enforce at least 6 pt for every figure label at final PDF placement,
  including mathematical subscripts, supplementary figures, and formerly
  outlined schematic labels.
- [x] Regenerate main figures, compare numerical reports with the baseline,
  and visually inspect source graphics and compiled figure pages.
- [x] Compile with resolved citations/references and no overfull boxes, verify
  statistics and fonts, and run the relevant production regression tests.

## Analysis identity

The selected completed reproduction is
`outputs/clean_production_reproduction_f7b7e5e_20260827`, analysis commit
`43264db`, checkpoint SHA-256
`dd6780a8b34a5a280adb3926fa1662b229743d1a3b49affaf1590721ef153540`.
Figure 3, Figure 4, and the new illustrative replay identify this checkpoint.
The original bundle was preserved; revised renders and replay products are
under `manuscript/build/`.

The empirical Figure 2 cache is
`outputs/dekel240_paper/m77_epoch279/production_figure3/cache/covdecomp_derived.pkl`,
SHA-256 `b9c058c9c3d99bc18826c1b4177c1f680af3eae6a4c815c70f546dea7b3a5fd2`.
It is also the selected Figure 3 empirical input. Its older directory label
does not select the digital twin checkpoint.

`analysis/figure_audit.json` records final PDF hashes, rendering-source hashes,
manuscript hashes, numerical source provenance, final font sizes, and equality
checks. `analysis/empirical_stats.json` preserves Figure 2's unrounded summaries;
`analysis/panel_a_selection.json` preserves the illustrative replay selection.

## Model and methods corrections

The selected model has **19,917,876 parameters**, takes **35 × 35 pixels** and
**60 frames at 240 Hz**, and is feedforward. The description now covers the
learned temporal filters, sign splitting, multiscale visual core, behavioral
encoding and gain, sparse readout, signed first-stage phase branch, and
softplus output. Figure 3's architecture diagram matches these components.

Architecture/settings were checked against the selected model YAML,
`VisionCore` implementation, loaded checkpoint, and executed curriculum
manifest under `CLEAN_REPRO_f7b7e5e_20260827_s201`. Training is 70/15/15 for
natural-stimulus train/validation/test splits, with 488 initial-training,
24 frozen-core, and 64 fine-tuning epochs. Optimizer, regularization,
sampling, and checkpoint selection are described as executed. The last
two stages expose all grating repeats; reported grating performance is
therefore descriptive. Fixed RSVP trials provide the held-out comparison.

The 8.33-ms comparison bin comes from summing native 240-Hz predicted counts
onto the fixed 120-Hz empirical grid. The same-bin affine calibration and
its optimistic evaluation are explicit. Normalized metrics retain values
above one when their denominator is positive. Figure 3E's finite [0,1]
restriction, condition-specific pairs, unit-level TOST mean-difference test,
±0.1 equivalence margin, and distinction between that test and the shaded
median reference band are now stated.

Figure 4 methods now distinguish the all-unit population from the strict
validation subset, describe tuning-validation thresholds and unconverged
finite fits, define the 55% response passband, specify the actual spectral
quartiles and bootstrap procedures, and interpret stage outputs as cumulative
trained-readout contributions. Behavioral input is zero in these visual
counterfactuals. Panel A is illustrative; panels B–H do not select epochs by
microsaccade detections.

## Numerical corrections

| Result | Current draft / source |
| --- | --- |
| Analysis-eligible recordings | 1,022 units across 19 sessions; 10–118 per session (Allen 876/11, Logan 146/8). |
| Figure 2C FEM fraction | Median 0.71, 95% CI [0.69, 0.72], 939 units. |
| Figure 2E Fano summary | Mean of session-specific slopes 1.25 → 0.81; replaces the pooled-slope values 1.41 → 0.82. |
| Figure 2F noise correlation | 0.076 → 0.010; difference −0.066, 95% CI [−0.084, −0.050]. |
| Figure 2I alignment | 0.61 ± 0.12 and 0.63 ± 0.14 (session mean ± SD). |
| Figure 3C | 987 units, 19 sessions; full/retinal/stabilized median CCnorm 0.648/0.645/0.560. |
| Figure 3D | 969 units; PSTH/full/retinal/stabilized medians 0.139/0.237/0.227/0.043; full improvement in 16/19 sessions. |
| Figure 3E | Empirical/full/retinal/stabilized medians 0.736/0.722/0.685/0.282; corrected equivalence and paired-test probabilities. |
| Figure 4 population | 725 units; 145 strictly validated units are used to select panel D examples. |
| Figure 4B | At median path 107.3 arcmin: rate +52.8% [44.2, 62.0], SSI +21.4% [14.8, 29.2]. |
| Figure 4G | Lowest → highest engagement bin: rate 3.7% → 65.1%; SSI 1.3% → 23.9%. |
| Figure 4H | 10 images × 10 traces = 100 movies. |

Other paired contrasts, confidence intervals, passband/path correlations,
and relevant probabilities are exported with the principal summaries into
78 macros by `sync_stats.py`.

**Removed unsupported Figure 2C significance claim:** the existing alpha
aggregation compares an observed pooled unit median with session-level
shuffle medians. Those are different aggregate statistics. The old pooled
`p = 0.007` claim is omitted; the median and unit-bootstrap interval remain.
No replacement pooled significance claim is made. Figure 2E/F/I retain the
matched aggregate shuffle calculations actually implemented. The underlying
alpha null code was not changed as part of manuscript preparation.

## Figure 4A

The image shortlist ranks central-half image-gradient energy before examining
neural responses. The same checkpoint then scores 12 images × 12 candidate
trace windows × 725 units = 104,400 combinations. Disclosed population-unit
preselection and bounded response/clarity gates produce image **56**, trace
**92**, endpoint **214**, unit **289**. The image has visible central vertical
structure; the response maps show change through the center of the field.

The selected stabilized/moving rates are **10.95/23.78 spikes/s**, and SSI is
**0.111/0.314 bits/spike**. Alignment uses the model's rounded mean first-layer
peak lag of 15 frames (62.5 ms). The Methods and selection record disclose that
the response-audited example is illustrative and separate from population
inference. Figure 4B–H summary dictionaries remain exactly equal to the
completed production reports.

## Typography and validation

| Figure | Final PDF page | Smallest measured text |
| --- | ---: | ---: |
| 1 | 3 | 6.154 pt |
| 2 | 5 | 6.096 pt |
| 3 | 8 | 6.500 pt |
| 4 | 12 | 6.100 pt |
| Supplement 1 | 38 | 6.087 pt |
| Supplement 2 | 40 | 6.149 pt |

All main figures were regenerated with text exported as fonts. Figure 1's
seven outlined schematic labels were restored from their SVG `aria-label`
metadata. Font floors account for mathematical subscripts. Figure 3/4 use
manuscript layouts, with captions moved to following pages to preserve usable
size. Supplement 1 retains the imported vector plots with text enlarged at
the existing baselines; supplement 2 is unchanged. Both were inspected.

After visual review, Figure 3B was redrawn as an illustrated architecture:
retinal movie, learned filter planes, three feature-map stacks, multiscale
combination, behavioral MLP and modulation, spatial/feature readouts, phase
bypass, and a real held-out prediction. The feature textures are outputs of
the selected checkpoint for the cached natural-image history, using the
production pixel normalization and lag order. Four channels per spatial stage
are selected by spatial variance and individually contrast-normalized for
display; readout glyphs are explicitly schematic. The caption distinguishes
these elements. `analysis/figure3_schematic.json` records the checkpoint,
history hash, selected channels, and example PSTH identity. Figure 3 now uses
the full text width; overlapping trace labels, readout routes, and statistical
annotations were also corrected. The final revision's 22 targeted core,
Figure 3 runner, and source-closure tests pass (`build/figure3_revision_tests.txt`).

- Final 40-page PDF: all six figures pass the 6-pt minimum and source text-bound
  checks; installed main PDFs match their audited render outputs.
- Figure 3C/D/E statistics and Figure 4B–H population reports: exact equality
  with the completed analysis, including the bootstrap reports.
- Production regression suite (31 test files added/changed from `origin/main`
  to `HEAD`): **206 passed**, recorded in `build/regression_tests.txt`.
- Final renderer, exemplar-selection, and source-closure checks:
  **20 passed**, recorded in `build/final_targeted_tests.txt`.
- The documented `empirical-stats`, `figures`, and `audit` Makefile targets
  were exercised end to end; final logs are under `build/final_*`.
- `git diff --check` passes. Final LaTeX/BibTeX logs have no unresolved
  citations/references, overfull boxes, or bibliography warnings. Shell escape
  remains disabled; its informational warning is expected.

The missing bibliography field for `kleiner2007s` was completed using
[Psychtoolbox's citation guidance](https://psychtoolbox.org/credits).

## Scope and interpretation

This audit establishes consistency with the selected executed analysis; it
does not retrain the model or acquire new empirical recordings. Figure 3's
same-bin calibration, unit-level equivalence inference, descriptive grating
assessment, and the response-selected Figure 4A example are explicitly
qualified in the text. The first supplement's scientific content is preserved
from the supplied draft because its generator was not included. Archived
drafts and commented-out legacy passages are retained as historical material;
they are not compiled into the active manuscript.
