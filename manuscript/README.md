# Manuscript

The supplied Overleaf export was imported on 2026-09-10. The active manuscript
uses the completed rank-one model without the separate first-stage readout.
Figures 3 and 4, their model-dependent statistics, feature-map illustrations,
and Methods use the same selected checkpoint. Figures 1 and 2 and the two
supplements retain their empirical analyses. See [POSTING_AUDIT.md](POSTING_AUDIT.md)
for the completed migration and earlier revision history.

`analysis/selected_model_bundle.json` pins the completed analysis and result
hashes. Figure 3B depicts the selected checkpoint's actual architecture;
`analysis/figure3_schematic.json` records the source and displayed architecture.

Figure 3B also depicts the input interventions with the same colors as C--E:
Full (blue), Retinal only (red; extraretinal input zeroed before the MLP), and
Stabilized (purple; retinal motion removed, extraretinal input retained).
Both retinal histories enter the same core; all conditions use the same weights.
Each shared input uses a single arrow. The paired cubes use the cached FixRSVP
illustrations, preserving image flashes, and feature textures are evaluated on
the displayed measured history. The caption distinguishes the illustration's
trial-medoid stabilization/shared current frame from the session-global gaze
anchor used in the quantified analysis. The eye traces also come from FixRSVP. `analysis/figure3_schematic.json` records this mapping.

Figure 4 uses continuously Gaussian-filtered eye trajectories (6-ms standard
deviation) with verified subdegree microsaccades. Its C/F filled contours compare
47 drift windows and 18 microsaccade-containing windows (<1 degree), with
unit dynamic power per trace and equal animal weights. All response analyses,
tuning, passband overlays, and stage contributions use the selected rank-one
model and corrected trajectory selection. The former interim panel override
and manuscript draft notes have been retired.

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

This requires the local `yatesfv` Conda environment and the selected empirical
cache. Methods settings and recording counts are documented in the audit;
the generated macros cover the principal reported Figure 2–4 results.

## Regenerate figures and audit the PDF

```sh
# Render from the selected analysis, including its exact Figure 4A replay.
make -C manuscript figures

# Or select individual figures.
conda run --no-capture-output -n yatesfv python manuscript/render_figures.py 3 4

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
`conda run --no-capture-output -n yatesfv python`.

The driver writes intermediate products to `build/` and installs final PDFs
under `figures/`. It leaves the completed analysis bundle intact. The first
supplement uses the imported vector graphic with enlarged text; its original
generator is not present in this export. The second supplement already meets
the font requirement and retains its imported PDF.

The audit needs the regenerated Figure 3/4 manifests and Figure 4A replay
summary under `build/`. It checks installed PDF identity, source text bounds,
all measurable figure text after LaTeX scaling (including math subscripts),
and exact agreement of Figure 3 numerical summaries and Figure 4B–H reports
with their selected sources. It saves `analysis/figure_audit.json` with
source/result hashes and final font sizes, and page previews under
`build/page_previews/`. Font checks complement visual inspection; after
changing layout, inspect the compiled figure pages for collisions as well.

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
