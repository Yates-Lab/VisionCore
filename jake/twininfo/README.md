# Twininfo

Concise production pipeline for the fixational-eye-movement information story.
All outputs are written under `outputs/twininfo/<analysis_name>/`. The default
folder name is human-readable and based on the scientific analysis settings, for
example `images-24-29-30_1crop_2fix-2ms_16units-per-pixel_cross-grid_128frames_seed-0`.
Use `--run-name` to force a specific folder.

## Main Command

Run a small real-data validation pass:

```bash
conda run --no-capture-output -n yatesfv python -m jake.twininfo.pipeline \
  --run-name validation_three_images \
  --image-indices 24 29 30 \
  --n-crops-per-image 1 \
  --n-examples-per-kind 2 \
  --population-size 16 \
  --shift-grid-mode cross \
  --make-stimulus-movies \
  --recompute
```

Run production image selection over every available natural image:

```bash
conda run --no-capture-output -n yatesfv python -m jake.twininfo.pipeline \
  --run-name production_all_images \
  --n-crops-per-image 3 \
  --n-examples-per-kind 10 \
  --population-size 100 \
  --make-stimulus-movies \
  --recompute
```

By default, the simulated population is the top `--population-size` biological
readout units per retinotopic pixel, ranked by Ryan's Fig. 3 normalized
correlation (`ccnorm`) after filtering to `ccmax > 0.80`. The information
calculation keeps the convolutional rate map and treats every selected unit at
every output grid position as the surrogate V1 population. To change this:

```bash
  --population-selection top_performance \
  --performance-metric ccnorm \
  --min-performance-score 0.5 \
  --population-grid-position-mode full_grid \
  --population-grid-stride 2 \
  --deduplicate-units \
  --dedupe-correlation-threshold 0.95 \
  --ccmax-threshold 0.80
```

Omit `--min-performance-score` to simply take the top `N` units after the
`ccmax` filter. You can also rank by single-trial bits/spike if Ryan's
`mcfarland_outputs_mono.pkl` cache is present:

```bash
  --performance-metric fixrsvp_bps
```

The de-duplication step runs a small shared stimulus battery through candidate
twins and greedily removes lower-ranked biological units whose center-grid
responses correlate above the threshold with a higher-ranked unit. This reduces
near-identical feature twins before gathering responses across space.

The selected units are written to `metadata/00_population_units.csv`.

Activation-map movies are slow and are off by default:

```bash
conda run --no-capture-output -n yatesfv python -m jake.twininfo.pipeline \
  --run-name validation_three_images_with_activations \
  --image-indices 24 29 30 \
  --n-crops-per-image 1 \
  --n-examples-per-kind 2 \
  --population-size 16 \
  --shift-grid-mode cross \
  --make-stimulus-movies \
  --make-activation-movies \
  --recompute
```

If `--image-indices` is omitted, every natural image in the FIXRsvp stack is
used. If `--recompute` is omitted and the same config already exists, the
pipeline returns the existing run summary.

## Code Map

1. Eye movement selection and visualization:
   `trace_selection.py`

2. Image selection, four-layer pyramid reconstruction, residuals, local
   pyramid phase scrambling, middle-band energy maps, and crop hotspot selection:
   `image_selection.py`

3. Retinal stimulus MP4s:
   `retinal_movies.py`

4. Activation-map MP4s:
   `activation_movies.py`

5. Individual cumulative information traces:
   `pipeline.py`, `run_information_step`

6. Spatial-frequency cumulative information traces:
   `pipeline.py`, `SF_CONDITIONS`

7. Average gain summaries relative to stabilized movies:
   `pipeline.py`, `_plot_gain_summary`

Core reusable helpers:

- `common.py`: shared model/data loading and constants from Ryan's `_common.py`
- `eye_controls.py`: microsaccade detection and trace controls
- `retinal_examples.py`: exact retinal crop rendering and pyramid image controls
- `lagcube_information.py`: model lag-cube rates, FI, spatial SSI
- `information.py`: Poisson Fisher and single-spike information math
- `stimuli.py`: natural-image loading and small image helpers

## Output Map

Each run has:

- `metadata/01_trace_examples*.csv`: selected fixation and microsaccade windows
- `metadata/00_population_units.csv`: selected model units, performance scores,
  and retinotopic grid positions
- `figures/01_*trace_selection*.pdf`: trace selector QC
- `metadata/02_image_crop_hotspots.csv`: selected crop centers and offsets
- `figures/02_image_selection_page_*.pdf`: image selector QC
- `movies/stimulus_*.mp4`: retinal stimulus MP4s, when enabled
- `movies/activation_maps_*.mp4`: activation-map MP4s, when enabled
- `metadata/05_lagcube_information_summary.csv`: final per-movie metrics
- `cache/cumulative_information_series.npz`: cumulative FI and additive
  spatial-information traces, with prefix-normalized spatial SSI saved
  separately
- `figures/05_*`: phase-control information traces
- `figures/06_*`: spatial-frequency information traces
- `figures/07_*`: final metric and gain summaries
