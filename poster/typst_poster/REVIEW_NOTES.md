# Poster review notes — 2026-11-09

Review of the poster as committed in `ad0f638`. The build reproduces the approved
reference exactly. The headline numbers match `manuscript/generated_stats.tex`:
1,022 units / 19 sessions, f_FEM median 0.71 (IQR 0.57–0.81, n = 939), Fano
1.25 → 0.81, corrected noise-correlation CI −0.010 to 0.029, and single-trial
fractions 0.139 / 0.235 / 0.226 / 0.042 (n = 962).

Panel numbers below use the **current** numbering, after the column-1 redesign.
Old → new: 1 and 3 retired, 2 → 2, 4 → 3, 5 → 4, 6 → 5, 7 → 6, 8 → 7, 9 → 8.

## Done (in the approved reference)

- **Column 1 redesign.** It is now an intro ("Fixational eye movements: noise or
  signal?"), panel 1 ("Gaze shifts within an RF change population responses",
  rebuilt from the R01 figure) and panel 2 (single unit, unchanged).
  - Old panel 1 (setup and dataset stats) and old panel 3 (gaze-grouped rasters)
    are retired.
  - Their content moved into the intro and new panel 1. That covers the rig, DPI
    image, dataset line, same trials and gaze-window shading.
  - The intro's Q1/Q2 point to panels 1–5 and 6–8.
- Semantics of new panel 1 checked against the raw data; see the docstring of
  `paper/fig1/generate_fem_overview.py`.
  - Gaze is (x right, y up). RFs are in retinal coordinates and drawn at each
    group's mean gaze.
  - The image placement reproduces the dataset's gaze-contingent crops.
  - The 0.33° shift is smaller than 99% of single-unit RFs (median diameter 0.42°).
- Panel titles enlarged from 36 pt to 40 pt.
- Header: "Tejas" changed to "Tejasvi" Kothapalli. The author line stays centred.
- Reference [4] changed to Ressmeyer 2026. The tracker paper is now published:
  J Neurosci Methods 429:110693, doi:10.1016/j.jneumeth.2026.110693 (PMID 41581702).
  `manuscript/refs.bib` (`ressmeyer2025openirisdpi`) still cites the 2025
  bioRxiv version.
- Funding: added "NIH/NEI T32-EY007043 (T.K.)". "Fuster Endowment (A.C.H.)" was
  already credited. The manuscript acknowledgements have the Fuster Endowment
  line but not the T32.
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
6. **Intro has spare space at the bottom.** It could be shortened in favour of
   panel 1.

## Likely questions at the poster (no change needed)

- Panel 6: why the Stabilized model (0.042) scores below the trial average (0.139).
- Title "explains cortical variability": this is backed by the P3/P4
  decomposition, while the model captures 23.5% of σ²_rate.
- Funding line credits R01-EY018363 to M.R. (Rucci, not an author). This matches
  the manuscript and is presumably intentional (DPI tracker support).

## Polish (low priority)

- Panels 4 and 7 have copper titles and the rest are navy. Confirm this is an
  intentional highlight.
- P3: "Trial 1" is above "Trial 2" in the eye-position plot but below it in the
  spike-rate plot.
- P8 heatmap: larger tick font than its neighbours, "log ₂" has a stray space,
  and the colorbar uses hyphen "-4" instead of a true minus.
- P2: the grey time-window lines and the pink line in "Trial fixations" are unexplained.
- P6: "append" label is crowded against the readout arrow.
- QR code: not machine-checked here; scan it to confirm it opens the preprint.
- Local clutter: the ignored `apply_*.py` scripts and `before_*` backups in this
  folder can be deleted.
