# Orientation-corrected Figure 4

Production spectral replay (`paper/fig4/spatiotemporal_tuning/spectral_power.frequency_grid`, `ky = -fy`) pairs image
orientation 180−θ with tuning orientation θ. This directory rebuilds every Figure 4 artifact that depends on
orientation-weighted spectral predictors, using the assay convention from
`declan/fig4_direction/corrected_orientation_replay.py`. Production code is not modified.

Affected: displayed panels F (engagement) and G (cumulative readouts; the top-passband movie selection changes),
the passband-vs-path-length statistic, the passband/normalized-overlap regressions, and the zero tests.
Panels A–E are unchanged (pixel-identical).

## Run order (from the repo root)

```bash
# 0. corrected predictors (already done 2026-09-25)
PYTHONPATH=/home/declan/DataYatesV1 .venv/bin/python declan/fig4_direction/corrected_orientation_replay.py
# 1. corrected copies of the two production shards
.venv/bin/python declan/fig4_orientation_fix/build_shards.py
# 2. rebuild with Jake's unmodified production scripts, pointed at the corrected shards (GPU)
declan/fig4_orientation_fix/rebuild_bundle.sh
# 3. passband regressions (control: --variant production must reproduce outputs/passband_comparison_20260914)
PYTHONPATH=. .venv/bin/python declan/fig4_orientation_fix/passband_comparison.py --variant corrected
# 4. zero tests, manuscript-layout PDF, generated_stats.tex, macro diff
PYTHONPATH=. .venv/bin/python declan/fig4_orientation_fix/manuscript_outputs.py
# 5. install
cp outputs/figures/fig4_orientation_fix/figure4.pdf manuscript/figures/figure4.pdf
cp outputs/stats/fig4_orientation_fix/source_root/manuscript_fix/generated_stats.tex manuscript/generated_stats.tex
```

Step 2 uses the release manifest's recorded arguments, with `--population-shards` / `--stage-trajectory` /
`--passband-comparison` swapped to `outputs/stats/fig4_orientation_fix/bundle/figure4/...`. Local substitutions of
unreadable files in Jake's tree, all hash-identical:
`response_replay_bank_.../image_feature_table.csv` → the bundle's `response_matrix_40img_x_200fix/merged` copy
(sha 404aa0…); the dataset config → this repo's copy (sha 98d6c2…); the Figure 2 cache → Ryan's copy (sha b9c058…).
Local scikit-learn gives different CV folds, so the regression reuses the released fold assignments.

## Controls

- `compare_passband_path_length.py` on production shards: identical to the released summary.
- `build_figure4.py` on production inputs: all 90 numeric summary values identical to the release.
- Passband regression on production shards: all 719 summary values identical; normalized overlap identical.
- Stage-trajectory replay: model identity checks match the release (cached response max error 1.3e-3 spikes/s).
- Release audit on corrected artifacts: `release_ready: true`; provenance binds.
- `sync_stats.py --check` against the release reproduces the current `manuscript/generated_stats.tex`.

## Outputs

- `outputs/figures/fig4_orientation_fix/figure4.pdf` — manuscript layout, drop-in for `manuscript/figures/figure4.pdf`
- `outputs/stats/fig4_orientation_fix/source_root/manuscript_fix/generated_stats.tex` — drop-in for `manuscript/generated_stats.tex`
- `outputs/stats/fig4_orientation_fix/macro_changes.csv` — every changed macro
- `outputs/stats/fig4_orientation_fix/bundle/figure4/` — rebuilt shards, trajectory, path-length stat, release figure/audit/provenance
- `outputs/stats/fig4_orientation_fix/passband_comparison_{production,corrected}/` — regressions (control and corrected)
