# Poster review notes — 2026-11-09

Review begun at `ad0f638`, updated for the completed panel-1 layout below. The
build reproduces the approved reference exactly. The headline numbers match
`manuscript/generated_stats.tex`:
1,022 units / 19 sessions, f_FEM median 0.71 (IQR 0.57–0.81, n = 939), Fano
1.25 → 0.81, corrected noise-correlation CI −0.010 to 0.029, and single-trial
fractions 0.139 / 0.235 / 0.226 / 0.042 (n = 962).

Panel numbers below use the **current** numbering, after the column-1 redesign.
Old → new: 1 and 3 retired, 2 → 2, 4 → 3, 5 → 4, 6 → 5, 7 → 6, 8 → 7, 9 → 8.

## Done (in the approved reference)

- **Panel 6 breathing room (2026-11-10):** the histogram heading stands above
  the median annotations, with its condition legend in the plot's blank upper
  left. The prediction strip uses three lines: its heading and the precise
  median fraction of conditional rate variance explained. Unit/session counts
  move to the existing methods line above. Plot dimensions, marks, values,
  medians and all other panels are unchanged. Pixel checks at 72 and 144 dpi
  restrict differences to these annotations; the legend covers no data.

- **Final clarity pass (2026-11-10):** retained the overall layout and intro.
  Panel 6 distinguishes reproduction of FEM-related rate variability from
  single-trial prediction performance, retaining the precise rate-variance
  metric beneath the new strip heading. Histogram legend keys move 14 pt down
  and median/band annotations 8 pt down; all plotted bins and median x-values
  are unchanged. Panel 7 attaches “Example unit and fixation” directly to the
  rate/SSI percentages. Panel 8 explicitly qualifies the result as model-based
  and explains tuning engagement as motion-generated retinal power falling
  within a unit's tuning, retaining within-unit percentiles. Panel 4's Poisson
  label now sits beside y=1.0. Panel 2 specifies the 50-ms sorting windows and
  notes the 21-ms initial onset segment: `paper/fig1/generate_fig1d.py` defines
  12 and 5 bins respectively at 240 Hz. These are code/source and preservation
  checks, not independent reproduction of the scientific analyses. Pixels
  outside the intended annotation areas are identical at 72 and 144 dpi.

- **Panel 1 spacing (2026-11-10):** setup notes use upright Arial at their
  existing sizes. Gaze strips move down 6 pt; the 306 pt square RF plot moves
  up 16 pt to accommodate a two-row legend within its width. The RF heading
  wraps at the comma. The lower legend, takeaway and response annotations
  share an 888 pt final baseline above the dataset strip. Wording, data and
  plot dimensions are preserved; other regions are pixel-identical at 72 and
  144 dpi.

- **Heading consistency (2026-11-09):** all eight panel headings use 48 pt
  Arial bold at x=59.75 pt, baseline=48.07 pt. Badges share a 44.26 pt diameter,
  center (29.14, 36.34) pt, and white 2.5 pt outline; 29 pt Arial bold numbers
  start at x=20.81 pt, baseline=45.82 pt. Panel 2's four plot headings now use
  bold weight at their existing sizes and positions; the RF citation moves
  right to retain clearance. Scientific wording and plot layouts are unchanged.

- **Panel 1 review fixes (2026-11-09):**
  - **Setup palette:** brain labels use the existing muted trial red; the hot
    mirror and its key use a softer cyan. Eye-inset labels are white, with one
    muted sand accent for outlines and arrows. Trial colors are unchanged.
  - **Right-edge clearance:** moved the gaze traces, rasters, population-rate
    plot and their scales 20 pt left as one group. Plot widths remain 332 pt;
    heights, marks, data and wording are unchanged. This increases clearance
    between the rotated scale labels and the panel border without shrinking plots.
  - **Gaze traces:** both strips share one angular scale, so the single 0.1° bar
    holds for both. Previously each strip scaled to its own range, and the vertical
    one was about 10% compressed.
  - **Eye-image labels:** "Pupil" and "CR" sit inside the pupil, clear of the ring.
  - **RF plot title:** now "Foveal RFs, all mapping sessions". It shows 30
    mapping sessions, distinct from the 19 recording sessions in the stats bar.
  - **Direct labels:** the first view labels its own marks: "gaze, 5 trials",
    "RF" and "RF on trials 6–10". The second view reads by analogy.
  - **Other group in each view:** its gaze is shown as faint dots in its own
    colour, and its RF as a dashed outline in that colour, pairing with the bold
    dots and solid outline of this view's trials.

- **Final reviewer fixes (authorized 2026-11-09):** panel 3 now shows Trial 1
  above Trial 2 in both the eye-position and spike-rate examples. Each spike
  trace and its label move together by the existing separation between the
  zero-rate baselines; path coordinates, timing, amplitudes, scale bars and
  highlighted time windows are preserved. Panel 6's title is now
  "Gaze-dependent retinal input predicts single trials", retaining the paper's
  small extraretinal contribution rather than categorically excluding it.

- **Panel 1 raster legibility.** Spike ticks are now 3 pt tall with 1.35 pt
  strokes (previously about 2.65 pt and 0.9 pt). All 4,465 spike centers and the
  source-trial order are preserved. Display rows are numbered 1–10 downward,
  with group headings "Trials 1–5" and "Trials 6–10". Panel 2's heavier raster
  marks remain unchanged. `make_print_proof.py` exports 300 dpi panel previews
  and actual-size vector PDF crops for a physical viewing-distance check;
  screen inspection does not establish printed legibility.

- Added superscript asterisks to Declan P. Rowley, Ryan A. Ressmeyer and
  Tejasvi Kothapalli, preserving affiliation numbers and centering the author
  line. "*Equal contribution" closes the last affiliation line, which stays
  centred.

- **Column 1 redesign.** It is now an intro ("Fixational eye movements: noise or
  signal?"), panel 1 ("Tiny gaze shifts reshape population responses",
  rebuilt from the R01 figure) and panel 2 (single unit, unchanged).
  - Old panel 1 (setup and dataset stats) and old panel 3 (gaze-grouped rasters)
    are retired.
  - Their content moved into the intro and new panel 1. That covers the rig, DPI
    image, dataset line and selected trials. The current rasters show only the
    gaze-grouping window, so they no longer need window shading.
  - The intro's Q1/Q2 point to panels 1–5 and 6–8.
- Semantics of new panel 1 checked against the raw data; see the docstring of
  `paper/fig1/generate_fem_overview.py`.
  - Gaze is (x right, y up). RFs are in retinal coordinates and drawn at each
    group's mean gaze.
  - The image placement reproduces the dataset's gaze-contingent crops.
  - The 0.13° shift is about a third of the median single-unit RF (0.42°).
  - **RF position comes from the fixation trials** (`_task_rf_offset`): 0.41° left,
    0.06° below gaze (95% CI y −0.11 to −0.02°). This is consistent with Figure 1C's
    across-session RFs. The separate gaborium mapping put it at 0.56° left, 0.10°
    up, so only the RF's shape and size come from the mapping. Re-cropping with
    mirrored gaze axes explains the residual spiking much less well, which rules
    out a coordinate flip.
- **Panel 1 layout C** puts one stimulus/RF view beside each group's rasters.
  Red and blue link the gaze traces, dots, RF outlines, trial labels and PSTHs.
  - The group with the higher mean vertical gaze is shown above the other group
    in both the poster and standalone R01 figure. A dashed outline shows the
    other group's RF position in each poster view.
  - The setup follows the supplied compact illustration: the vector brain sits
    above the marmoset, a cyan outlined hot mirror lies between the eye and screen,
    and the camera sits below it. A red dashed eye-position line and cyan mirror
    key explain the optical schematic. The 110 pt DPI inset tucks beside the
    animal, with its labels on a separate PDF layer and the tracker note below.
  - Removed the dashed connectors from the screen to the enlarged stimulus view;
    the screen-to-eye lines within the rig remain.
  - The all-session RF plot shows both animals' mapping outlines, this session's
    population RF, and the example's 0.13° gaze shift at the same angular scale.
    Its plotting area is 306 pt square (previously 278 pt), with outline opacity
    raised from 0.06 to 0.10. The 20 pt legend has more clearance below the frame.
    This is RF context, not an all-session analysis
    of gaze spread versus RF diameter.
  - Trial and gaze labels are 21 pt. The population-response takeaway uses 24 pt
    semibold navy beside the PSTHs. A bottom strip gives the dataset statistics.
    The body is shifted 68 pt right so the setup and RF plot align with panel 2's
    90 pt artwork inset; the title and badge stay fixed. Trace, raster and PSTH
    widths are 332 pt (previously 352 pt) to keep the right-hand scale labels
    inside the panel. Their heights, text sizes, data and stimulus-view dimensions
    are unchanged, as is the intro's space.
  - Liberation Sans Italic 2.1.5 is bundled for the call-outs (metrically
    identical to Arial; same open licence as the other Liberation fonts).
  - Build, overlap/boundary audit, approved-reference validation and clean-clone
    repeat builds pass. The DPI repair preserves the supplied data and artwork;
    all existing fields in the example cache match the preceding committed cache.
- **Panel 8 heatmap typography:** ticks and axis labels reduced from 23.52 to
  21 pt and changed from black to the neighboring plots' dark gray. Text anchors
  preserve tick alignment and centered axis labels. Wording and graphics are
  unchanged.
- **Panel 1 trials:** Figure 1F's grouping search on the 258–392 ms window, with
  cluster-centre distance capped at 0.06° (`SEGMENT_START`,
  `GROUP_CENTROID_DISTANCE` in `generate_fem_overview.py`). The first-ranked
  grouping whose trials all have data across the window is used.
  - Red trials 5, 20, 25, 26, 74; blue trials 24, 43, 67, 84, 87. All five per
    group are shown.
  - Mean gaze shift 0.13° (about a third of an RF).
  - The response is reshaped rather than scaled: red is higher early in the
    window and blue later.
  - Rasters and PSTH cover exactly the grouping window. The background is image 8,
    on screen one response lag before the largest group difference.
  - Chosen from six candidates near 0.13° (108–392 ms windows). The 158–292 ms
    example (1.5× rate ratio) was the alternative.
- **Panel titles:** 48 pt, rewritten as claims.
  1. Tiny gaze shifts reshape population responses
  2. One neuron’s “noise” tracks the eye
  3. Eye movements drive most single-cell rate variance (71%)
  4. Correcting for gaze removes shared noise
  5. Eye-driven variability follows stimulus dimensions
  6. Gaze-dependent retinal input predicts single trials
  7. In the model, eye motion sharpens spatial coding
  8. Sharpening grows with tuning and model depth

  The intro title is unchanged, at 48 pt.
- Header: "Tejas" changed to "Tejasvi" Kothapalli. The author line stays centred.
- Reference [4] changed to Ressmeyer 2026. The tracker paper is now published:
  J Neurosci Methods 429:110693, doi:10.1016/j.jneumeth.2026.110693 (PMID 41581702).
  `manuscript/refs.bib` (`ressmeyer2025openirisdpi`) still cites the 2025
  bioRxiv version.
- Funding: added "NIH/NEI T32-EY007043 (T.K.)". "Fuster Endowment (A.C.H.)" was
  already credited. The manuscript acknowledgements have the Fuster Endowment
  line but not the T32.
- Footer: "Acknowledgements" heads a line beginning "Additional preprint
  coauthors:" for Dekel Galor, Ruei-Jr Wu and Michele Rucci, the authors in
  `manuscript/main.tex` absent from the poster's header. Below it, the paper's
  thank-you sentence credits Yonatan Abrham (Mitchell Lab) for collecting some
  of the data. Funding remains in its existing block rather than being repeated.
  Contact details and the existing author line are unchanged.
- Panel 3 title specifies "single-cell rate variance" at the existing 48 pt size.
- Panel 3's separate large "71%" is removed; the value stays in the title. The
  histogram is 30 pt taller (460 pt) with its baseline fixed. A 22 pt-wide navy
  downward triangle replaces the thin median tick at the identical x coordinate.
  Histogram bin paths, counts, limits and scientific definitions are unchanged.
- **Intro (2026-11-09, for SfN):** text only. Navy headline "What looks like
  noise may reflect active sensing" with "noise" and "active sensing" in copper;
  a paragraph on response variability across repeated presentations and the
  sensory effects of the animal's own movements; a separate line, "Even during
  fixation, small eye movements continually reshape visual input."; and two
  copper-barred questions ("How much cortical variability comes from the eye?" /
  "Do fixational eye movements help encode spatial detail?"). No methods line or
  illustration: panel 1 shows the setup and data directly below. Text aligns with
  panel 1's title. Opens on a problem every SfN attendee knows and hedges as the
  paper does ("often", "can include", "may reflect"). The previous
  "noise, or sensing?" version with an eye schematic and raster is in 1851d8f.
- Panel 2:
  - The "RF projection axis" leader is darker and thicker and ends on the axis.
  - The (0, 0) marker is a small plain black cross.

## Open — worth considering

1. **Unit counts change between panels without explanation:** 1,022 (dataset line),
   132 (panel 1), 939 (P3), 962 (P6), 725 (P8). P3 and P6 both plot "FEM fraction
   of rate variance". P6's score is normalized by the Fig 2 conditional rate
   variance (n = 939) but covers 962 units. A one-line inclusion footnote would
   cover it. The same 939 vs 962 gap is in the manuscript.
2. **Panel 7's example unit is far stronger than typical, and the panel doesn't say so.**
   It shows Rate +199% and SSI +188%, near the display-gate ceilings of 200%.
   The population effects are +23.2% rate and +9.0% SSI (FigFourRateEffect /
   FigFourSSIEffect). Consider adding the population numbers to the panel.
3. **Panel 8 says "All quintiles/stages > 0", but the S1 stage plots at about 0.000.**
   Consider marking which stages are significant directly on the plot.
4. **"Conditional rate variance" is never defined in words.** σ²_rate = σ²_PSTH +
   σ²_FEM, so it is not conditional on the stimulus. A short definition would help.
5. **Panel 1 is one session.** Its "shift < RF size" example could be backed by a
   19-session summary comparing fixational gaze spread with RF diameter.
6. **Intro** is text only and fills its space with evenly distributed blocks. It
   could be shortened in favour of panel 1 (requires refitting panel 1's height).
7. **Standalone R01 figure** (`generate_fem_overview.py`) shares panel 1's trials,
   RF placement and image, but keeps the original R01 layout.

## Likely questions at the poster (no change needed)

- Panel 6: why the Stabilized model (0.042) scores below the trial average (0.139).
- Title "explains cortical variability": this is backed by the P3/P4
  decomposition, while the model captures 23.5% of σ²_rate.
- Funding line credits R01-EY018363 to M.R. (Rucci, not an author). This matches
  the manuscript and is presumably intentional (DPI tracker support).

## Polish (low priority)

- Panels 4 and 7 have copper titles and the rest are navy. Confirm this is an
  intentional highlight.
- P8 heatmap: "log ₂" has a stray space, and the colorbar uses hyphen "-4"
  instead of a true minus.
- P2: the grey time-window lines and the pink line in "Trial fixations" are unexplained.
- P6: "append" label is crowded against the readout arrow.
- QR code: not machine-checked here; scan it to confirm it opens the preprint.
- Local clutter: the ignored `apply_*.py` scripts and `before_*` backups in this
  folder can be deleted.
