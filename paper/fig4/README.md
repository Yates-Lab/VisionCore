# Figure 4

Compose the figure:

```bash
uv run python paper/fig4/generate_figure4.py
```

That works today and needs nothing from this README. Everything below is about
*regenerating the cached data* the figure composes from, which does not yet work
end to end.

## State of reproducibility — read this first

The figure renders from 36 `fig4_*` files in `outputs/cache/`. Those files are
**untracked and gitignored**. They are the only copy of this figure's data.

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

## What blocks full reproduction

Four stages are `BLOCKED` in the preflight. All four block on data that exists
only under `/home/declan`, which is mode 700:

| Stage | Needs |
|---|---|
| `merge_ssi_shards` | `.../backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/shards` (169 MB). Confirmed present. Without it the merged bank is the earliest reproducible point. |
| `instantaneous_unit_maps` | `backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1/` and `backimage_axis_conditioned_matched_static_percandidate_gpu1_n128_c4_k16_scales_0p5_1_2_bconsistent_v1/` |
| `sf_group_ssi_modulation` | `backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/` (74 MB) |
| `schematic_final_maps` | `backimage_rr100_instantaneous_unit_maps_latest_v1/` (597 MB) — i.e. the output of `instantaneous_unit_maps` |

The two run directories for `instantaneous_unit_maps` are **not** the
similarly-named directories sitting beside them. They are the values recorded
in `cache_identity_json` inside the shipped `fig4_unit_maps.npz`, which is the
only statement of what actually produced that cache. The plausible-by-name
alternative (`..._sf_contour_alignment_long_axis30_...`) postdates it.

One producer is still unidentified: `fig4_trace_bank_metadata_filtered.csv`,
from `backimage_trace_bank_diffusion_large_fixation_sample_n5000_n40_v1/filtered_path_length_le350arcmin/`.
Five recovered scripts read it; none writes it.

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
