# Figure 4

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
uv run python paper/fig4/stage_cache_overlay.py /home/declan/VisionCore
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
honest now: on a clean checkout `refresh_all.py` reports 14 blocked stages
because the lower upstream trees and merged trace bank are absent. Those blocked
stages collapse to these source boundaries:

| Boundary | State |
|---|---|
| Real-trace SSI matrix scorer | The canonical launcher is now in repo (`upstream/run_real_trace_matrix.py`) and records the recovered 100 image x 1000 trace production profile, but the scorer body is still not ported from the recovered `declan` tree. |
| Real-trace SSI matrix merge | In repo (`upstream/merge_backimage_real_trace_ssi_matrix_shards.py`). It needs generated shard dirs `.../backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/shards/images_000_050` and `images_050_100`. |
| Upstream fixation-window data | `window_features.csv` is a true upstream input outside this compact Fig. 4 module (`sha256 e8e2fa28c39d4d0222502bbe73fc221210260212fbed25bdc6c2e6c6217f73ba`, 76,832 rows, 57 columns). |
| RR100/model assets | The recovered production checkpoint is `/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints/learned_resnet_none_convgru_gaussian_ddp_bs128_ds30_lr1e-3_wd1e-4_corelrscale.5_warmup5/epoch=147-val_bps_overall=0.5702.ckpt` (`sha256 55d084aa0beb7d65614aecb9122edf7ad49c5799d370dbbd5dcf60b815c62de3`). The RR100 population spec hashes are recorded in the launcher. |
| Other RR100 panel producers | `instantaneous_unit_maps`, `sf_group_ssi_modulation`, and `schematic_final_maps` still need lower output trees under `outputs/active_sensing_movie_information/` and `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/`. |

The two run directories for `instantaneous_unit_maps` are **not** the
similarly-named directories sitting beside them. They are the values recorded
in `cache_identity_json` inside the shipped `fig4_unit_maps.npz`, which is the
only statement of what actually produced that cache. The plausible-by-name
alternative (`..._sf_contour_alignment_long_axis30_...`) postdates it.

One producer is still unidentified: `fig4_trace_bank_metadata_filtered.csv`,
from `backimage_trace_bank_diffusion_large_fixation_sample_n5000_n40_v1/filtered_path_length_le350arcmin/`.
Five recovered scripts read it; none writes it.

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

Use the smoke profile for the same schemas at tiny scale:

```bash
uv run python paper/fig4/upstream/run_real_trace_matrix.py --profile smoke
```

Until the scorer body is ported into this repo, execution requires naming the
recovered runner explicitly:

```bash
FIG4_REAL_TRACE_MATRIX_RUNNER=/path/to/run_backimage_real_trace_ssi_matrix_pilot.py \
FIG4_STABILIZED_BASELINE_RUNNER=/path/to/run_backimage_real_trace_stabilized_baseline.py \
FIG4_RR100_POPULATION_SPEC_DIR=/path/to/step1_activation_fingerprints \
uv run python paper/fig4/upstream/run_real_trace_matrix.py --run-all --force
```

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

This pin is verified to *load the model*. It has **not** been shown to reproduce
Declan's numbers; that needs a regenerated RR100 output diffed against baseline.
`schematic_final_maps` is the shallowest stage that would test it — it reads one
directory and runs on `cuda:1`.

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

Upstream producer scripts are located via `FIG4_RECOVERED_ROOT`, defaulting to
`/home/ryanress/declan_recovery/VisionCore/declan`. On Declan's machine set it
to `/home/declan/VisionCore/declan`.

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
are read by `refresh/_fig4_geometry_story.py`, are absent from the cache, and are
declared in `REFRESH_ONLY_INPUTS` rather than `REQUIRED_INPUTS`. The compose path
never imports that module, so requiring them would make `generate_figure4.py`
refuse to render a figure it can in fact render.
