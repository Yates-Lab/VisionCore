# Figure 4

> **Production note:** this file documents the legacy cache-first Figure 4
> pipeline inherited from `main`. The native-240 manuscript production path is
> [`spatiotemporal_tuning/FIGURE4_PRODUCTION.md`](spatiotemporal_tuning/FIGURE4_PRODUCTION.md),
> with the repository-wide execution order in
> [`../../PRODUCTION_PIPELINE_HANDOFF.md`](../../PRODUCTION_PIPELINE_HANDOFF.md).
> Do not mix legacy RR100 caches or the 120-Hz checkpoint described below with
> the native-240 production artifacts.

Compose the figure:

```bash
uv run python paper/fig4/generate_figure4.py
```

This is cache-first. The command fails loudly if the required caches are not in
`outputs/cache/`; use `--allow-missing` only for a visibly degraded layout smoke
render.

## State of reproducibility — read this first

The compose path preflights 25 required `fig4_*` files in `outputs/cache/`.
Those files are **untracked and gitignored**. `REFRESH_SOURCES` also names
refresh-only/support artifacts so the producer graph can account for everything
without making the compose path stricter than the figure actually needs.

Reproducibility is partial, and the parts are not equally solid:

| Status | What it means | Which caches |
|---|---|---|
| **Regeneration-verified** | Producer was rerun into a scratch tree and its output diffed clean against the shipped cache | Panels B, C, D, E, F, G, H — the path-bins, bracket, story, edge-coherence, patch-radius, gallery, behaviour-window and bridge caches |
| **Provenance-confirmed** | Cache is byte-identical (md5) to a file in the producer's own output directory, but the producer has **not** been rerun | The 4 `fig4_unit_maps*`, `fig4_sf_tuning_unit_groups.csv`, the 3 `fig4_schematic_*`, `fig4_trace_component_movie_metrics.csv`, `fig4_image_feature_table.csv`, `fig4_trace_xy.npy` |
| **Not regenerated, by design** | Hand-tuned or writes into `CACHE_DIR` | `fig4_panel_{a,d}_layout_overrides.json`, `fig4_panel_a_network_icon.pdf` + its provenance JSON |

Provenance-confirmed is a weaker claim than it sounds. It proves the cache was
not hand-edited on its way out of the producer's output directory. It says
nothing about whether running that producer today gives the same numbers.

**Caveat on the regeneration-verified row.** Those 20 verdicts were recorded
before the producer mapping was corrected (see below). Six mappings changed
since, so the verdicts should be re-established with `--verify` before anyone
relies on them. Nothing suggests they will fail; they simply have not been
re-run against the corrected graph.

## Staging the handoff caches

The historical handoff archive keeps Declan's original output paths; the current
figure reads flat `outputs/cache/fig4_*` names. Stage through the explicit mapper
rather than extracting the archive into this repo:

```bash
uv run python paper/fig4/stage_cache_overlay.py /path/to/ssi_figure_v4_cache_overlay.tar.gz
```

The compact cache tarball supplies 16 of the 25 compose-required inputs. The
remaining nine are lower-root or raw-data products: the four RR100 unit-map
caches, merged real-trace `image_feature_table.csv`/`trace_xy.npy`,
contour-relative trace component movie metrics, RR100 SF tuning groups, and the
cached schematic stimulus payload.

If you have the old full output tree available, stage from that root instead:

```bash
uv run python paper/fig4/stage_cache_overlay.py /path/to/full/VisionCore-output-tree
```

When a sibling `DataYatesV1` checkout is available next to that source tree, the
stager also builds `fig4_schematic_stimulus_payload.npz`. Use
`--data-package-root PATH` if the data package lives elsewhere. Use `--dry-run`
to audit the mapping first and `--strict` when a missing required cache should
be treated as a failed staging step. A JSON manifest with copied files, hashes,
and remaining missing required inputs is written beside the staged caches.

The full-tree stage is the reference path today: it supplies the CI-bearing
path-bin cache and producer-schema edge-coherence files needed for an exact
match to `reference/figure4_reference.pdf`.

## Building a clean cache bundle

Once `outputs/cache/` is staged, package the current flat caches into a portable
bundle:

```bash
uv run python paper/fig4/build_cache_bundle.py
```

Verify the bundle without touching the live cache:

```bash
uv run python paper/fig4/verify_cache_bundle.py outputs/figures/fig4/handoff/fig4_cache_bundle_YYYYMMDD.tar.gz
```

The verifier unpacks into a scratch directory, points `VISIONCORE_CACHE_DIR` and
`VISIONCORE_FIGURES_DIR` there, composes Fig. 4, and requires a zero pixel
difference against `reference/figure4_reference.pdf`.

## What blocks full reproduction

Cache-first composition is reproducible from the bundle. Source refresh is more
honest now: in the staged working tree used for this recovery, `refresh_all.py`
reports 12 blocked stages because lower upstream trees and the flat staged
merged trace bank are absent. Those blocked stages collapse to these source
boundaries:

| Boundary | State |
|---|---|
| Real-trace SSI matrix scorer | In repo (`upstream/score_real_trace_matrix.py`, with shared code in `upstream/real_trace_matrix/`). It records model/checkpoint/readout/RR100 provenance at runtime and keeps the recovered 100 image x 1000 trace production profile in the launcher. |
| Real-trace SSI matrix merge | In repo (`upstream/merge_backimage_real_trace_ssi_matrix_shards.py`). It needs generated shard dirs `.../backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/shards/images_000_050` and `images_050_100`. |
| Upstream fixation-window data | `window_features.csv` is a staged source asset at `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/window_features.csv` (`sha256 e8e2fa28c39d4d0222502bbe73fc221210260212fbed25bdc6c2e6c6217f73ba`, 76,832 rows, 57 columns). |
| McFarland readout artifact | The source scorer needs `scripts/mcfarland_outputs_mono.pkl` or `scripts/mcfarland_outputs.pkl` to rebuild the canonical 756-channel readout. Override with `FIG4_MCFARLAND_OUTPUTS` or `--mcfarland-outputs`. |
| RR100/model assets | The recovered production checkpoint is staged locally at `outputs/artifacts/model_checkpoints/fig4_twin/epoch=147-val_bps_overall=0.5702.ckpt` (`sha256 55d084aa0beb7d65614aecb9122edf7ad49c5799d370dbbd5dcf60b815c62de3`). The RR100 population spec hashes are recorded in the launcher. |
| Other RR100 panel producers | `instantaneous_unit_maps`, `sf_group_ssi_modulation`, and `schematic_final_maps` still need lower output trees under `outputs/active_sensing_movie_information/` and `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/`. |

The two run directories for `instantaneous_unit_maps` are **not** the
similarly-named directories sitting beside them. They are the values recorded
in `cache_identity_json` inside the shipped `fig4_unit_maps.npz`, which is the
only statement of what actually produced that cache. The plausible-by-name
alternative (`..._sf_contour_alignment_long_axis30_...`) postdates it.

`fig4_trace_bank_metadata_filtered.csv` is now recovered as an in-repo
metadata-only upstream build:

```bash
uv run python paper/fig4/upstream/build_trace_bank_metadata.py --force
```

The default contract matches the historical diagnostic cache: sample 5000
reviewed BackImage/FEM windows with pandas `random_state=20260716`, sort by
`source_row`, center-crop native 40-sample traces, then write the
`path_length_arcmin <= 350` filtered subset.

## Deep real-trace matrix launcher

The recovered production command is now captured by a safe-by-default launcher:

```bash
uv run python paper/fig4/upstream/run_real_trace_matrix.py --profile production
```

That writes `outputs/figures/fig4/provenance/real_trace_matrix_production_plan.json`
and prints the shard, merge, and stabilized-baseline commands without running
the 20-hour scorer. It explicitly records the 32-frame model-history versus
40-scored-sample analysis boundary, checkpoint identity, pinned dataset config,
RR100 population spec hashes, direct inputs, and expected outputs.

Use the smoke profile for the same schemas at tiny scale. It keeps the full
source CSV as the input contract, then filters to `Allen_2022-02-16` before
sampling one image and two traces:

```bash
uv run python paper/fig4/upstream/run_real_trace_matrix.py --profile smoke
```

Execution uses in-repo scorer scripts by default. `--run-all` still refuses on a
clean checkout until the source inputs and model/readout/RR100 artifacts are
present:

```bash
FIG4_MCFARLAND_OUTPUTS=/path/to/mcfarland_outputs_mono.pkl \
FIG4_RR100_POPULATION_SPEC_DIR=/path/to/step1_activation_fingerprints \
uv run python paper/fig4/upstream/run_real_trace_matrix.py --run-all --force
```

To stage those assets from a VisionCore-style data tree without importing any
code from it:

```bash
# inspect only
uv run python paper/fig4/upstream/stage_real_trace_source_assets.py /path/to/VisionCore

# local validation without duplicating the 2.5 GB McFarland pickle
uv run python paper/fig4/upstream/stage_real_trace_source_assets.py \
  /path/to/VisionCore \
  --apply \
  --link-mode symlink

# portable local asset tree
uv run python paper/fig4/upstream/stage_real_trace_source_assets.py \
  /path/to/VisionCore \
  --apply \
  --link-mode copy
```

The stager verifies the exact hashes for `backimage_image_fem_windows.csv`
(`ac2364e22ede162940a9ba7de5c8ab2c2ef2ea9c9985d851e302769d17c12567`),
`dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv`
(`7a506b617ccbda563cab1e7f10173f9015448f88ce71a1abec7b05dc8aaa92f2`), and the
two RR100 population spec files. It records the McFarland pickle path and size
by default; pass `--hash-large` if you want to hash that 2.5 GB file too.

To turn a staged source-asset tree into a portable handoff bundle, build a
separate source bundle:

```bash
uv run python paper/fig4/upstream/build_real_trace_source_asset_bundle.py
```

The source bundle is distinct from the flat cache bundle: it packages the
reviewed image/FEM source table, raw fixation-window table, RR100 unit metadata,
RR100 population spec, and McFarland readout artifact at the paths expected by
the in-repo launcher. It
dereferences locally staged symlinks, so `stage_real_trace_source_assets.py
--link-mode symlink` can be used for validation before making a portable
archive. The recovered model checkpoint is intentionally excluded unless
`--include-checkpoint` is passed. When included, it extracts to
`outputs/artifacts/model_checkpoints/fig4_twin/`, which the launcher
auto-detects unless `FIG4_TWIN_CHECKPOINT` or `--checkpoint-path` is set.

Install the source bundle into another clean checkout with:

```bash
tar -xzf outputs/figures/fig4/handoff/fig4_real_trace_source_assets_YYYYMMDD.tar.gz \
  -C /path/to/VisionCoreMain
```

The raw-data trace-bank step also requires the optional `DataYatesV1` data
package/environment. The repo records that dependency in `pyproject.toml` under
the `data` extra, but the large data files are intentionally not git-tracked.
When local checkouts are used instead of installing the extra, expose them with
`PYTHONPATH=/path/to/DataYatesV1:/path/to/DataRowleyV1V2`.

After staging, the end-to-end source smoke is:

```bash
uv run python paper/fig4/upstream/run_real_trace_matrix.py \
  --profile smoke \
  --run-all \
  --force
```

The recovered production sampler has also been checked against the historical
matrix cache: with seed `20260717`, the clean source path selects the same 100
image rows, the same 1000 trace source rows, and the same 1000 trace hashes.
A one-image/full-trace-bank production shard (`--only-shard 0:1`) then diffed
cleanly against the old merged cache at `atol=1e-5, rtol=1e-5`, with zero
array differences for SSI, expected spikes, mean rate, and population SSI.
The full 100-image production matrix still needs the normal two long shard runs.

### Temporal burn-in contract for the refreshed matrix

Each refreshed image x trace movie contains 72 native FEM samples at 120 Hz:
32 explicit history samples followed by the historical central 40 scored
samples. For a 128-sample source window these are source samples `12:84` and
`44:84`, respectively. The 72-frame trace is embedded directly, without the
historical `trace[:32] + trace` prefix seeding.

Embedding 32 lags from 72 frames produces 41 outputs. The first output, whose
current frame is model-trace frame 31, is discarded. SSI, expected spikes, and
mean rate accumulate only outputs 1--40, whose current frames are 32--71. With
0-based scored sample `s` and lag channel `l` (`l=0` is current), the model-trace
index is therefore `32 + s - l`.

`trace_xy.npy` remains the 40-sample scored trace consumed by downstream Figure
4 path, RMS, microsaccade, and binning analyses. `trace_xy_model.npy` stores the
72-frame burn-in-plus-scored model input. The trace-bank builder computes all
movement features and sampling strata from the scored array only. Stabilized
baselines use a 72-frame zero-motion input and the identical 40-output mask.

To audit the clean scorer against a historical matrix cache without resampling
images or traces, replay the selected tables and `trace_xy.npy` from that cache.
The audit launcher plans by default and executes with `--run`:

```bash
uv run python paper/fig4/upstream/audit_real_trace_matrix_replay.py \
  --reference-dir /path/to/backimage_real_trace_ssi_matrix.../merged \
  --image-start 0 \
  --image-stop 1 \
  --trace-start 0 \
  --trace-stop 1000 \
  --device cuda:1 \
  --run \
  --force
```

This replay mode is an equivalence audit, not a source-regeneration shortcut:
it fixes the historical image/trace selections and checks that the clean scorer,
model pin, RR100 population view, and matrix row contract reproduce the old
cache rows. For a faster spot check, use `--image-stop 5 --trace-stop 100`;
for the full trace-bank contract on one historical image, use
`--image-stop 1 --trace-stop 1000`.

For a single production shard:

```bash
uv run python paper/fig4/upstream/run_real_trace_matrix.py \
  --only-shard 0:50 \
  --run-shards \
  --force
```

The merge step itself is standalone in this repo:

```bash
uv run python paper/fig4/upstream/merge_backimage_real_trace_ssi_matrix_shards.py \
  --out-dir DIR/merged \
  DIR/shards/images_000_050 \
  DIR/shards/images_050_100
```

## The model pin — `upstream/dataset_configs/`

The RR100 stages need the digital twin, and loading it from the checkpoint
fails on a stock checkout. `MultiDatasetModel` sizes each readout from
`len(cids)` read live off disk, the checkpoint stores no cids of its own, and
the session configs it points at have drifted since it was trained — all 20
readouts mismatch.

`upstream/dataset_configs/` is a frozen copy of `experiments/dataset_configs/`
at VisionCore **`e6c85ae`** (2026-02-01), the revision whose `cids` match the
checkpoint. Verified: readouts load at `strict=True` with shapes
`[120, 156, 139, 169, 71, 80, 164, 119, 124, 71, 156, 137, 163, 75, 51, 133, 182, 168, 69, 65]`,
and the canonical selection comes to 756 units — the number `CanonicalTwinScorer`
documents.

Override with `FIG4_DATASET_CONFIGS` if you want a different pin.

This pin is verified to load the model and to reproduce the real-trace matrix
for a one-image/full-trace-bank production slice. Other RR100 panel producers
still need their own regenerated-output diffs against baseline.
`schematic_final_maps` is the shallowest remaining stage that would test those
non-matrix RR100 paths; it reads one directory and runs on `cuda:1`.

## Running the refresh

```bash
# preflight: what is runnable, what is blocked, and why
uv run python paper/fig4/refresh_all.py

# regenerate into a scratch tree (never into outputs/cache/)
uv run python paper/fig4/refresh_all.py --run --scratch DIR [--only k1,k2]

# diff a scratch tree against a baseline copy of the caches
uv run python paper/fig4/refresh_all.py --verify --scratch DIR --baseline DIR

# recompute the numbers quoted in the manuscript
uv run python paper/fig4/refresh_all.py --manuscript-numbers --cache-dir DIR
```

Legacy recovered producer scripts are located via `FIG4_RECOVERED_ROOT`. By
default the runner looks for an optional local overlay at
`paper/fig4/upstream/recovered_legacy/`; set `FIG4_RECOVERED_ROOT` only if you
intentionally want to run an external recovered script tree. The cache-first
figure and the in-repo real-trace matrix pipeline do not require that overlay.

Verdicts: `PASS`, `PASS-PROVENANCE` (differs only in recorded paths), 
`PASS-STALE-CACHE` (regenerated file is a strict superset), `FAIL`,
`NOT-REGENERATED`, `SKIP-BINARY`, `NO-OUT-DIR`, `REFUSED`. `NOT-REGENERATED` is
reported explicitly so a partial refresh can never read as a passing one.

## Traps

These cost real time to find. Preserve the behaviour that guards them.

- **`refresh_all.py` never writes to `CACHE_DIR`.** It refuses any stage whose
  out-dir is inside it (only `_fig4_network_icon.py` trips this). Regenerate
  into a scratch tree and diff. Back the cache up first; it is untracked.
- **`patch_radius_sensitivity` reads *and* writes its `--root`,** and the seed
  directory already contains the file it produces. Staging that unchanged would
  be a false PASS. The runner deletes a stage's own outputs after seeding.
- **Wrong `produces` filenames stage nothing and still look successful.** This
  was real: the bracket stage silently staged zero files, and six more mappings
  were wrong the same way. Always confirm a stage actually rewrote its outputs
  (mtime), not merely that it exited 0.
- **`fig4_a.pkl` and `fig4_bottomrow_ablation.pkl` are not ours.** They belong
  to `ryan/digital-twin-fem/` and `ryan/fig4/` — a name collision in the flat
  cache namespace. Do not delete them as orphans.

## Corrected producer mapping

`REFRESH_SOURCES` had six wrong attributions. All six corrections below are
md5-confirmed against the producer's own output, not inferred from names:

| Cache | Was credited to | Actually produced by |
|---|---|---|
| `fig4_unit_maps*` (4) | `run_backimage_contour_axis_rr100_spatial_ssi.py` | `plot_backimage_rr100_instantaneous_unit_maps.py` |
| `fig4_trace_component_movie_metrics.csv` | `analyze_..._phase1_phase2.py` | `analyze_backimage_contour_matched_trace_components.py` |
| `fig4_sf_tuning_unit_groups.csv` | `run_backimage_rr100_frequency_tuning_probe.py` | `plot_backimage_rr100_sf_group_ssi_modulation.py` |
| `fig4_edge_coherence_*` (2) | `plot_backimage_contour_motion_components.py` | `generate_backimage_contour_position_spread_followups.py` |
| `fig4_schematic_trace_center40.csv` | `compute_schematic_rr100_final_maps.py` | `make_ssi_contour_schematic.py` |

The `produces` filenames for the RR100 and schematic stages were also wrong —
e.g. `unit_maps.npz` for what is really
`cache/backimage_rr100_instantaneous_unit_maps.npz` — so those stages would have
staged zero files and still reported success.

Panel H had two undeclared prerequisites, `screen_backimage_local_feature_eye_metric_poles.py`
and `screen_backimage_spatial_frequency_eye_metric_scaling.py`. Both write into
the directory `seed_from` already seeds, which is why the stage passed with two
invisible dependencies. Their tables are now declared inputs.

`fig4_phase1_movie_analysis_table.csv` and `fig4_trace_bank_metadata_filtered.csv`
are read by `refresh/_fig4_geometry_story.py`, are absent from the compose cache,
and are declared in `REFRESH_ONLY_INPUTS` rather than `REQUIRED_INPUTS`. The
compose path never imports that module, so requiring them would make
`generate_figure4.py` refuse to render a figure it can in fact render. The
trace-bank metadata input can be regenerated with
`paper/fig4/upstream/build_trace_bank_metadata.py`; the phase1 movie-analysis
table is still regenerated through the phase1/phase2 real-trace matrix analysis
path.
