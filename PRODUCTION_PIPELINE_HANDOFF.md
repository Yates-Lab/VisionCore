# Native-240 model and Figures 3--4 production handoff

This branch is a clean, reviewable extraction of the native-240 model-fitting
pipeline and the production analyses for Figures 3 and 4. It is based on
`origin/main` at `7e7587f` and is intentionally **not merged into main**.

## Branch map

- `codex/development-p240c1-fig3-fig4` at `3d45ad5` is the source-only archive
  of the full analysis-development state. It is retained for archaeology, not
  as the production entry point.
- `codex/p240c1-production-pipeline` is this minimal production branch.
- The production branch contains no checkpoints, generated response arrays,
  figure caches, or unrelated data. Those remain external artifacts and are
  identified by path and SHA-256 digest in run manifests.

All Python entry points below must be run from the repository root in the
`yatesfv` conda environment. Commands that execute work use
`conda run --no-capture-output -n yatesfv`; dry runs may omit
`--no-capture-output`.

## Single source of model identity

[`paper/model_selection/production_model.yaml`](paper/model_selection/production_model.yaml)
is the only production model selector. It binds the retained checkpoint,
checkpoint digest, training manifest, three-stage curriculum, native-240
dataset configurations, architecture, parameter audit, and claim boundaries.
Downstream scripts resolve the model from this file and verify its hashes.

To inspect the contract without running a figure:

```bash
conda run -n yatesfv python paper/model_selection/audit_production_model.py \
  --spec paper/model_selection/production_model.yaml \
  --output /path/to/external/audit_production_model.json \
  --verbose
```

Changing models means changing this YAML deliberately and rebuilding every
downstream artifact. Do not change a checkpoint path in an individual figure
script, infer units by array position, or reuse a cache with a different
digest.

## 1. Three-stage native-240 fitting

The curriculum and handoff policies are documented in
[`experiments/curricula/README.md`](experiments/curricula/README.md). The three
stages are:

1. train the visual core, behavioral pathway, and ordinary readout;
2. freeze the shared representation and fit the sparse phase/readout branches;
3. unfreeze the model and fine-tune end to end with the core learning more
   slowly than the readouts.

Plan a run first:

```bash
conda run -n yatesfv python training/run_three_stage_curriculum.py \
  --run-id RUN_NAME
```

Execute on a chosen GPU:

```bash
conda run --no-capture-output -n yatesfv \
  python training/run_three_stage_curriculum.py \
  --run-id RUN_NAME --gpu 0 --execute
```

The runner writes an external manifest with resolved commands, source/config
hashes, stage status, checkpoint handoffs, and checkpoint hashes. It resumes an
interrupted stage from that stage's `last.ckpt`; `--start-stage` and
`--stop-after-stage` support intentional partial runs. Optimizer state never
leaks across stage boundaries.

## 2. Figure 3

The manuscript-facing entry point is
[`paper/model_selection/run_production_figure3.py`](paper/model_selection/run_production_figure3.py).
It requires the canonical observation and covariance artifacts explicitly,
hashes them and the analysis source closure, regenerates the model and
ablation caches, audits the ablation cache, and renders the PDF and PNG.

Inspect the resolved job without executing it:

```bash
conda run -n yatesfv python paper/model_selection/run_production_figure3.py \
  --canonical-observation-cache /path/to/canonical_observations.pkl \
  --covdecomp-cache /path/to/covariance_cache.pkl \
  --covdecomp-derived-cache /path/to/covariance_derived.pkl \
  --covdecomp-aligned-cache /path/to/covariance_aligned.pkl \
  --output-root /path/to/external/figure3_run \
  --gpu 0 --dry-run
```

Remove `--dry-run` and use `conda run --no-capture-output -n yatesfv` for the
production execution. `--reuse-existing-caches` is permitted only when the
inputs match the manifest. The completed run must contain `run_manifest.json`,
the cache audit, and hashed `figure3.pdf` and `figure3.png` artifacts.

Supporting audits include fixed-RSVP evaluation and trace validation
(`evaluate_fixrsvp.py`, `audit_fixrsvp_trace_cache.py`), the Figure 3
ablation-cache audit, and the real grating-tuning comparison. Their outputs are
supporting artifacts, not implicit inputs substituted by the renderer.

## 3. Figure 4

The scientific and release contract is
[`paper/fig4/spatiotemporal_tuning/FIGURE4_PRODUCTION.md`](paper/fig4/spatiotemporal_tuning/FIGURE4_PRODUCTION.md).
That document is authoritative for panel definitions, normalization, unit
inclusion, and claim boundaries. The dependency order is:

1. Measure exact-identity native-240 drifting-grating tuning at the primary and
   repeat contrasts with `run_exact_cid_drifting_tuning.py`; fail closed with
   `audit_exact_cid_drifting_tuning.py`.
2. Build the strict tuning contract with
   `build_exact_cid_figure4_contract.py`, then the explicitly labeled all-unit
   analysis population with `build_all_available_population_spec.py`. Unit
   identity is `(session, cid)` throughout; RR pooling, reclustering, and unit
   substitution are forbidden.
3. Build the zero-phase-filtered 240-Hz fixation bank with
   `build_real_fixation_bank.py`.
4. Plan and execute the exact-unit response matrix with
   [`paper/fig4/upstream/run_real_trace_matrix.py`](paper/fig4/upstream/run_real_trace_matrix.py):

   ```bash
   conda run -n yatesfv python paper/fig4/upstream/run_real_trace_matrix.py \
     --replay-matrix-dir /path/to/filtered_replay_bank \
     --unit-table /path/to/all_unit_tuning.csv \
     --population-spec-dir /path/to/population_contract \
     --population-version VERSION \
     --output-root /path/to/external/response_matrix \
     --profile release
   ```

   The default is a dry run. Add `--execute` under
   `conda run --no-capture-output -n yatesfv` after inspecting the manifest.
   The release profile is 40 images by 200 filtered traces, sharded in groups
   of 10 images. It uses 60 frames (250 ms) at native 240 Hz.
5. Audit/select Panel A examples and reduce the all-unit Panel B population
   with `audit_panel_a_exemplars.py` and
   `build_panel_b_population_path_length.py`.
6. Build the equal-dynamic-mass Kuang/Rucci spectra and exact spectral replay
   with `build_rucci_ensemble_power.py` and
   `build_matrix_spectral_replay.py`. The matched passband-versus-path-length
   text statistic comes from `compare_passband_path_length.py`.
7. Trace top-passband movies through the actual trained cumulative readout with
   `analyze_top_passband_stage_trajectory.py`. Release requires at least 100
   crossed movies; smaller results are smoke tests only.
8. Assemble the locked A--H layout with
   `run_production_figure4.py`. This runner is dry-run by default, verifies and
   hashes every explicit input, invokes the fail-closed release audit, and
   writes numerical result provenance only after every release gate passes.

The final runner's required artifact arguments are its chain of custody; view
them with:

```bash
conda run -n yatesfv python \
  paper/fig4/spatiotemporal_tuning/run_production_figure4.py --help
```

Use `--mode smoke --execute` only for layout/integration testing. A manuscript
artifact must use `--mode release --execute` and must report a passing
`audit_revised_figure4_release.py` result.

## Population and measurement boundaries

- Panel B uses all 725 checkpoint-available exact readouts and does not depend
  on tuning validation.
- The two Panel D examples come from the 145-unit strict validation subset.
- Panels E, G, and H use the declared all-725 exact-identity analysis
  population. The 580 strict-tuning failures remain disclosed; they are never
  called validated.
- Eye traces use the audited zero-phase 20-Hz-passband/30-Hz-stopband IIR and
  are sampled at native 240 Hz.
- Stabilization is anchored at the mean first-layer peak lag (15 frames,
  62.5 ms), so measured and stabilized activation maps are spatially aligned.
- Conditional spectra have equal TF>0 mass and separately audited complete
  power conservation.
- Panel G is a descriptive within-unit association, not variance explained or
  a fraction of the total effect.
- Panel H uses only the trained model's cumulative branches. No affine,
  tangent, ablated, or synthetic reference is part of the analysis.

## Validation and current release status

The focused cross-pipeline test suite is:

```bash
conda run -n yatesfv python -m pytest -q \
  tests/test_native240_model_pipeline.py \
  tests/test_three_stage_curriculum.py \
  tests/test_production_figure3_runner.py \
  tests/test_fixrsvp_evaluation.py \
  tests/test_fixrsvp_trace_audit.py \
  tests/test_fig3_ablation_cache_audit.py \
  tests/test_exact_cid_drifting_tuning.py \
  tests/test_exact_cid_figure4_contract.py \
  tests/test_eye_trace_filter_audit.py \
  tests/test_real_fixation_bank.py \
  tests/test_fig4_real_trace_matrix_plan.py \
  tests/test_panel_a_exemplar_audit.py \
  tests/test_matrix_spectral_replay.py \
  tests/test_passband_path_length_comparison.py \
  tests/test_top_passband_stage_trajectory.py \
  tests/test_figure4_renderer.py \
  tests/test_run_production_figure4.py
```

At handoff:

- the three-stage pipeline and model audit are complete;
- Figure 3 exactly reproduces the accepted production render from its declared
  external inputs;
- Figure 4 passes its real all-725 integration smoke except for the deliberate
  production-size gate: Panel H currently contains 40 crossed movies, while a
  release requires at least 100;
- the 40-image by 200-trace response-matrix release job has been successfully
  planned against the real checkpoint, native-240 dataset, exact 725-unit
  population, and filtered replay bank.

Therefore the Figure 4 source pipeline is handed off, but the final Figure 4
must not be described as release-ready until Panel H is rerun at 100 or more
crossed movies and the release audit passes.

## Repository hygiene

Generated outputs belong outside source control. Before review or handoff:

```bash
git diff --check origin/main...HEAD
git diff --name-only origin/main...HEAD
git status --short
```

Reject any newly tracked checkpoint, cache, response matrix, NumPy/pickle
artifact, generated figure, or unrelated data. Source manifests may record
absolute external paths because their SHA-256 digests are the identity check;
those artifacts themselves must remain untracked.
