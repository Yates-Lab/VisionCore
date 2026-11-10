# Reproducible scientific poster

This directory builds the approved 60 × 36 inch poster with editable Typst text,
retained scientific vector/image assets, and Matplotlib plots regenerated from
saved numerical results. It contains every required scientific and layout input.
No historical handoff directory, analysis environment, model checkpoints or
machine-specific paths are needed.

## First build from a clean checkout

Prerequisites: Git and **uv 0.9.25**. Python **3.12.12**, all 13 Python packages
(including Typst), and their distribution hashes are pinned in `pyproject.toml`,
`.python-version` and `uv.lock`. No separate Typst executable is required.
Run from the repository root:

```bash
cd poster/typst_poster
uv sync --frozen
uv run --frozen python setup_fonts.py --from-dir "$POSTER_LICENSED_FONT_DIR"
uv run --frozen python build.py
uv run --frozen python validate.py
```

Set `POSTER_LICENSED_FONT_DIR` to a directory containing your licensed copies of:

| File | Required font version |
| --- | --- |
| `arial.ttf` | Arial Regular 7.06 |
| `arialbd.ttf` | Arial Bold 7.06 |
| `times.ttf` | Times New Roman Regular 7.12 |

Arial supplies the main typography; Times New Roman supplies a fallback symbol.
The setup command checks the **exact SHA-256 hashes** in `fonts/manifest.json`
before copying these files into the project's Git-ignored `.fonts/` directory.
The source directory is used only during setup; the build never refers back to it.
A different release with the same family name can alter spacing, so missing or
mismatched fonts stop the build with instructions rather than trigger a silent
substitution. Obtain these fonts through your licensed installation; this project
does not distribute or download them. Preserve your own licensed originals for
future rebuilds. No font licensing rights are granted by this repository.

DejaVu Sans 2.37 and Liberation Sans 2.1.5 (regular, bold and italic; the italic sets panel 1's call-outs) are bundled in `fonts/open/` with their
license notices. Both renderers use the checked local fonts. Typst system-font
lookup is disabled. Subsequent builds need no font setup:

```bash
uv run --frozen python build.py --compare
uv run --frozen python validate.py
```

Initial dependency installation needs access to the package index (and Python
provider if Python 3.12.12 is not installed). Once provisioned, generation and
validation are offline; `uv run --frozen --offline ...` uses the locked local
installation/cache. Commands also work from other directories with
`uv run --project poster/typst_poster --frozen python poster/typst_poster/build.py`.
The original repository-wide `.venv` is not required.

## Outputs

- `output/poster_typst_full_60x36.pdf`: vector print master, 60 × 36 inches.
- `output/poster_typst_full.png`: full 4320 × 2592 pixel preview (72 pixels per
  physical inch at poster size; use the PDF for printing).
- `output/poster_typst_overview.png`: compact 1728 × 1037 preview.
- One crop per region in `poster_content.json` (`intro.png`, `panel_fem.png`,
  `panel2.png`, `panel4.png` … `panel9.png`), plus header and footer crops.
- `output/validation.json` and `output/layout_audit.json`: validation reports.
- With `--compare`: `output/approved_reference.png` and `output/difference.png`.
  An all-black difference image means no changed pixels.

Generated files, local fonts, virtual environments and historical editing
backups are ignored by Git. The nested ignore files explicitly retain required
JSON, NPZ and reference PDF inputs despite the repository-wide extension rules.

## Sources and scientific scope

- `poster.typ` and `poster_content.json`: page layout, scientific wording,
  positions, font sizes, colors and rotations. Text remains native/selectable.
  Region names keep the original panel numbers. Since the 2026-11-09 column-1
  redesign, the printed badge of region `panelN` (N ≥ 4) reads N − 1: the first
  column is now the intro, panel 1 (`panel_fem`) and panel 2.
- `assets/*.svg`: approved scientific graphics, including embedded raster data,
  clipping, original plotted paths and the model schematic (region `panel8`).
  These are editable graphics, not screenshots of complete panels.
- `components/panel_fem/`: the intro and panel 1 ("Tiny gaze shifts reshape
  population responses"). `plot_data.npz` holds the frozen example data, including
  the per-trial gaze traces. `rf_all_sessions.npz` holds the Figure 1C RF outlines
  for both animals. `render_plots.py` draws the panel-1 plots and rig;
  `intro.typ` composes the text-only intro ("What looks like noise may reflect active sensing"). `panel_fem.typ` overlays the
  vector `assets/marmoset.svg` and `assets/brain_v1.svg` on the Matplotlib layer.
  A separately rendered `assets/panel_fem_eye.pdf` places the annotated eye inset
  above the animal drawing, keeping the compact setup's overlapping layers clean.
  Both drawings come from the retired panel-1 artwork; the brain's V1 and fovea
  labels are outlines. `assets/dpi_eye.png` is the DPI eye image, with its labels
  drawn separately by Matplotlib. `export_data.py` documents provenance:
  it regenerates the data from `paper/fig1/generate_fem_overview.py` (same
  session, trials and RFs as the standalone R01 figure), needs the analysis
  environment, and is never run by the build. Its `--all-session-rfs` option
  exports the all-session RF cache from `paper/fig1/generate_fig1c.py` and the
  session STA caches. Ordinary builds need neither those caches nor raw data.
- `components/panel5/plot_data.npz`: unchanged matrix, Fano and correlation data
  used to regenerate panel 5. `render_plots.py` renders the plots at their approved
  physical sizes; `panel5.typ` composes the panel with native text.
- `components/panel5/plot_helpers.py`: the relevant drawing functions copied
  from `paper/fig2/_panel_common.py`, `generate_panel_fano.py`, and
  `generate_panel_noisecorr.py`. Implicit analysis-loading fallbacks were removed;
  the plotting operations are preserved. This isolates the build from future
  manuscript-helper changes and avoids importing the analysis dependency tree.
- `reference/approved.pdf`: the currently approved full poster, last updated
  for the 2026-11-09 review edits and column-1 redesign (see `REVIEW_NOTES.md`). `reference/inputs.json`
  records the corresponding hashes of the scientific/layout sources, saved data
  and reference PDF.

Reproduction starts from these approved assets and saved results. It does not
rerun model fitting or regenerate every panel from the experimental raw data.
The old one-time migration/edit scripts and historical exports are not part of
this build and must not be used to rebuild the current layout.

## Validation and a clean-checkout test

For print inspection after building, run `uv run --frozen python make_print_proof.py`
from this directory. It exports 300 dpi previews of panels 1 and 2 and
`output/raster_print_proof.pdf`: four Letter-landscape crops at the poster's exact
physical scale, preserving the source PDF graphics. Print at **100% / Actual size**
and check the one-inch calibration line before assessing viewing distance.
The full vector PDF remains the poster print master; the overview PNG is for screens.
Panel 1 uses 3 pt-tall spike ticks with 1.35 pt strokes. Displayed trials run 1–10
downward across the two groups; this numbering does not change source-trial order.

`validate.py` checks source/data hashes, exact extracted text, all used fonts
embedded, page dimensions, PNG/PDF agreement, panel crops, and **zero changed
pixels against the approved reference at both 72 and 144 dpi**. It also runs the
text-overlap/boundary audit. The rendering regression was verified on Linux
x86-64 with the pinned environment. Other platforms must pass the same checks;
matching package versions alone does not guarantee identical rasterization.
Validation intentionally fails after changes to the approved design or data;
review such changes before deliberately updating the reference. Normal builds
never refresh the reference or its hashes.

To repeat the isolation test from this checkout:

```bash
uv run --frozen python check_clean_build.py --offline
```

This selects only Git-visible poster sources (including new files awaiting
commit), creates a temporary fixture repository and a real clean clone, installs
a fresh environment from the lock, verifies missing/mismatched fonts are rejected,
provisions the licensed fonts from this checkout's already checked `.fonts/`,
and builds/validates from an unrelated working directory. It then rebuilds and
checks byte-for-byte stable PDF and PNG outputs. Temporary repositories are
removed on success or failure; your checkout's Git index/history is not changed.
Omit `--offline` if the locked packages are not cached. The report is written to
`output/clean_build_validation.json`.

Fixed export timestamps and an SVG hash salt remove time/random identifiers
from normal exports. The approved historical PDF has different metadata; visual
identity, native text and input integrity are the comparison with that reference.
Two new builds are also required to be byte-identical by the isolation test.
