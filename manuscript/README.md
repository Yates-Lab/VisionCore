# Manuscript

The supplied Overleaf export was imported on 2026-09-10. The active manuscript
uses the completed rank-one model without the separate first-stage readout.
Figures 3 and 4, their model-dependent statistics, feature-map illustrations,
and Methods use the same selected checkpoint. Figures 1 and 2 and the first two
supplements retain their empirical analyses. A third supplement compares
session-global, trial-centroid, and history-local retinal stabilization in the selected model. See [POSTING_AUDIT.md](POSTING_AUDIT.md)
for the completed migration and earlier revision history.

`analysis/selected_model_bundle.json` pins the completed analysis and result
hashes. Figure 3B depicts the selected checkpoint's actual architecture;
`analysis/figure3_schematic.json` records the source and displayed architecture.

Figure 3B also depicts the input interventions with the same colors as C--E:
Full (blue), Retinal only (red; extraretinal input zeroed before the MLP), and
Stabilized (ochre; retinal motion removed, extraretinal input retained).
Both retinal histories enter the same core; all conditions use the same weights.
Each shared input uses a single arrow. The paired cubes now use an unaltered
recorded FixRSVP history and a counterpart frozen at the session-global gaze centroid,
with image flashes preserved and a shared display contrast scale. Selection
uses retinal contrast and gaze displacement, independently of neural effects.
Both cubes preserve the actual flashed images without frame replacement.
Feature maps are recomputed from the displayed measured history. The schematic
uses the same global stabilization as the main quantified ablation; the
history-local control appears in the supplement.
The eye traces also come from FixRSVP. `analysis/figure3_schematic.json` records
this mapping and the selection of the held-out prediction example.

Figure 4 uses continuously Gaussian-filtered eye trajectories (6-ms standard
deviation) with verified subdegree microsaccades. Its C/D filled contours compare
47 drift windows and 18 microsaccade-containing windows (<1 degree), with
unit dynamic power per trace and equal animal weights. All response analyses,
tuning, passband overlays, and stage contributions use the selected rank-one
model and corrected trajectory selection. The former interim panel override
and manuscript draft notes have been retired. Individual tuning heatmaps are
omitted from the manuscript; their fitted contours overlay both spectra in C
and the power ratio in D. The population occupancy in E has no example contours.
Population passband occupancy is E, the power ratio is D, engagement is F, and
cumulative readouts are G. Source reports retain their original semantic keys
for numerical comparison; `display_panel_letters` records the mapping.

Gray boxes in B, F, and G show variation across units (IQR and median, with
5th--95th percentile whiskers). Colored points and capped bars show population
estimates and 95% bootstrap confidence intervals. F uses separate vertical
ranges that include the displayed unit-distribution whiskers. Annotations test
each population effect against zero improvement using its existing paired
bootstrap scheme, with Holm correction across all 32 displayed comparisons.
The retained draws and probabilities are in `analysis/figure4_zero_tests/`;
regenerate with `uv run --project .. --no-sync python manuscript/export_figure4_zero_tests.py`.

A and B share the top row, with B's rate and SSI axes stacked. A gives more
space to the complete response maps, displayed on one shared linear color
scale; its difference map uses symmetric limits. Horizontal arrows connect
each retinal movie directly to its activation map; the caption specifies the
shared model. D and E use wider axes with reduced side margins.
The illustration is selected
by the largest absolute SSI gain within the original top-eight-unit and
bounded response criteria. The eight leading distinct image--unit candidates
were replayed and visually inspected. The winner (image 56, trace 153, unit 376)
increases SSI from 0.269 to 0.485 bits/spike and rate from 10.4 to 28.0 spikes/s.
The updated illustration does not change any population estimates.

`analysis/figure4_example_selection.json` binds the retained example archive,
summary, and shortlist in `analysis/figure4_example/`. Reproduce the bounded
review and install its highest-ranked candidate with:

```sh
uv run --project .. --no-sync python manuscript/review_figure4_examples.py
uv run --project .. --no-sync python manuscript/review_figure4_examples.py --install-reviewed
uv run --project .. --no-sync python manuscript/render_figures.py 4
```

## Build the PDF

From the repository root:

```sh
make -C manuscript
```

The output is `manuscript/build/main.pdf`; the current page count and PDF hash
are recorded in `analysis/figure_audit.json`.
`latexmk` runs pdfLaTeX, BibTeX, and reference-resolution passes with shell escape
disabled. The PDF, logs, SyncTeX, and intermediate products stay in ignored
`build/`. The normal PDF build uses the bundled figure PDFs and generated
statistics; it does not require neural data, a checkpoint, Python, or a GPU.

Dependencies: GNU Make, `latexmk`, pdfLaTeX, BibTeX, and the packages loaded by
`lite.cls` and `main.tex`. On Debian/Ubuntu these are provided by `make`,
`latexmk`, `texlive-latex-extra`, and `texlive-fonts-recommended`.

```sh
make -C manuscript clean
```

This removes LaTeX build products, retaining the figure replay caches.

## Use an external artifact tree

Set the source root when the selected bundle is stored in another VisionCore
checkout. The scripts read artifacts from that tree and do not write to it.
Manuscript products stay under this checkout's `manuscript/build/`,
`manuscript/figures/`, and `manuscript/analysis/` directories. Figure 3 may
refresh derived caches under this checkout's `outputs/cache/`.

```sh
export VISIONCORE_MANUSCRIPT_SOURCE_ROOT=/home/jake/repos/VisionCore
```

The recorded Figure 2 cache is not readable in that checkout on this machine.
Point the workflow at the existing local cache directory; `sync_stats.py`
checks its recorded SHA-256 before using it.

```sh
export VISIONCORE_MANUSCRIPT_EMPIRICAL_CACHE_DIR="$PWD/outputs/cache"
```

Keep both variables set for statistics and Figure 3 rendering. Figure 4 uses
the selected external bundle directly. The manuscript selection remains
`analysis/selected_model_bundle.json`; repository-wide production defaults do
not replace it.

## Update statistics

```sh
make -C manuscript stats
make -C manuscript check-stats
```

`sync_stats.py` exports numerical summaries and model counts as LaTeX macros
into `generated_stats.tex`, which is
included by `main.tex`. It verifies the completed run, checkpoint identity,
Figure 4 source hashes, architecture counts, and empirical cache hash.
`check-stats` fails if the
macros have drifted from those inputs. Both commands write source hashes and
numerical source references to `build/stats_sources.json`.

Figure 2's exact summaries are stored in `analysis/empirical_stats.json`.
To recompute them through the current Figure 2 code and refresh the macros:

```sh
make -C manuscript empirical-stats
```

This requires the workspace uv environment and the selected empirical cache.
Methods settings and recording counts are documented in the audit;
the generated macros cover the principal reported Figure 2–4 results.

## Regenerate figures and audit the PDF

```sh
# Render from the selected analysis, including its exact Figure 4A replay.
make -C manuscript figures

# Or select individual figures.
uv run --project .. --no-sync python manuscript/render_figures.py 3 4

# Verify generated numbers, compile, and audit the installed figures.
make -C manuscript audit
```

Figure regeneration needs the repository's scientific environment, data and
analysis artifacts on the local mounts, and the Figure 1 SVG/PDF toolchain.
Figure 3 reads the selected checkpoint on CPU to illustrate its feature maps.
Figure 4 copies the exact illustrative replay from the selected completed
bundle; rendering does not select a new example. A fresh checkout needs that
bundle and its referenced local inputs to regenerate the figures. `SCI_PYTHON`
can override the Makefile's default
`uv run --project ../.. --no-sync python` command.

The driver writes intermediate products to `build/` and installs final PDFs
under `figures/`. It leaves the completed analysis bundle intact. The first
supplement uses the imported vector graphic with enlarged text; its original
generator is not present in this export. The second supplement already meets
the font requirement and retains its imported PDF.

The audit needs the regenerated Figure 3/4 manifests and Figure 4A replay
summary under `build/`. It checks installed PDF identity, source text bounds,
all measurable figure text after LaTeX scaling (including math subscripts),
and exact agreement of Figure 3 numerical summaries and the original Figure
4B–H source reports (allowing only C's contour-display flag). It also checks
the stabilization supplement's checkpoint, data identity, paired-score hashes,
and generated `stabilization_stats.tex`. It saves `analysis/figure_audit.json` with
source/result hashes and final font sizes, and page previews under
`build/page_previews/`. Font checks complement visual inspection; after
changing layout, inspect the compiled figure pages for collisions as well.

## Stabilization controls

The history-local replay fixes gaze within each 60-frame history at its latest
input position. The trial replay fixes each source frame at the centroid of its
own trial, using valid samples within 0.5 degrees and falling back to valid
fixation samples within 1 degree. Both controls preserve the recorded RSVP
sequence and use the Figure 3 C/D scoring pipeline, population, affine
calibration, correlation ceilings, and variance denominators. Neither replay
estimates a new FEM variance decomposition.

Both controls preserve temporal modulation from the 20-Hz flashed images.
The supplement also compares the within-unit costs of removing recent retinal
motion (Full minus History-local) and explicit extraretinal input (Full minus
Retinal-only). Their paired difference is Retinal-only minus History-local;
its session-bootstrap interval is computed directly, not from a difference of
population medians. The single-trial comparison does not establish a larger
cost for recent-motion removal (median 0.00077, 95% CI -0.00622 to 0.01316).

The completed replay is separate from the pinned production bundle:
`outputs/stabilization_control_20260915/`. Compact paired scores and summaries
are retained in `analysis/stabilization_control/`; the summary is bound by
`analysis/stabilization_control.json`. Re-render without model inference:

```sh
uv run --project .. --no-sync python manuscript/render_figures.py stabilization
```

For a trial-centroid replay, use a new output directory and a CUDA-enabled
environment. The renderer accepts the retained history-local cache separately.

```sh
VISIONCORE_MANUSCRIPT_SOURCE_ROOT=/path/to/source \
VISIONCORE_MANUSCRIPT_EMPIRICAL_CACHE_DIR="$PWD/outputs/cache" \
uv run --project .. --no-sync python paper/fig3/run_history_stabilization.py \
  --out-dir outputs/trial_stabilization_control --reference trial_centroid --gpu 0
uv run --project .. --no-sync python manuscript/render_stabilization_control.py \
  --inference-dir /path/to/source/outputs/stabilization_control_20260915 \
  --trial-inference-dir outputs/trial_stabilization_control
```

Each replay resumes from a per-session partial cache. The history-local replay
checks every current frame against the recorded retinal image.
`test_history_stabilization.py` checks history-local image identities and the
trial-centroid primary and fallback anchors against the native renderer without
running the model.

## Selected analysis and files

- Model/analysis bundle: `outputs/no_phase_readout_comparison_20260910/rank1`.
- Checkpoint SHA-256:
  `e70f287462155607a96c7d23d9d56e347111319aa48cafd3f54fe88dd4484cb9`.
- Empirical input: `outputs/dekel240_paper/m77_epoch279/production_figure3/cache/covdecomp_derived.pkl`,
  SHA-256 `b9c058c9c3d99bc18826c1b4177c1f680af3eae6a4c815c70f546dea7b3a5fd2`.
- `main.tex`, `refs.bib`, `lite.cls`: active manuscript and bibliography/layout.
- `generated_stats.tex`: generated numerical results; update through the exporter.
- `analysis/`: small, retained numerical summaries, example selection, and audit.
  `figure3_schematic.json` records the real feature-map textures and the
  held-out PSTH used in the model illustration.
- `figures/`: active PDFs and retained Overleaf alternatives.
- `old_figures/`, `main-backup0805.tex`: retained historical inputs, including
  the second supplementary PDF still included in the active draft.

`analysis/selected_model_bundle.json` selects the completed analysis bundle
and expected checkpoint for statistics, rendering, and the figure audit.
Regenerate the numerical summaries and figures and repeat the
manuscript/code audit after changing the selection. Repository-wide defaults
may point to an earlier model; they do not determine this draft's analysis. The broader
pipeline is described in `../PRODUCTION_PIPELINE_HANDOFF.md`.
