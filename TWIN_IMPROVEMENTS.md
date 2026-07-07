# Twin Loading & Training — Improvement Backlog

Running list of digital-twin fragilities and planned improvements. We append
line items here as we hit them, then do a consolidated refactor + retraining
run once there's time for another training cycle and polish.

---

## 1. Readout-size mismatch when loading a twin (config drift)

### Failure mode

Loading a checkpoint fails with a wall of
`size mismatch for model.readouts.N.{bias,mean,std,theta,features.weight}`
errors — the checkpoint's per-session readout has, e.g., 120 units but the
freshly-built model has 116.

Root cause: **readout sizes are not stored in the checkpoint.** At load time
`MultiDatasetModel.__init__` calls `load_dataset_configs(cfg_dir)` and rebuilds
each readout head to `n_units = len(dataset_config['cids'])`
(`models/modules/models.py:402-404`). The `cids` lists live in mutable per-session
YAMLs (`experiments/dataset_configs/sessions/<session>.yaml`). The checkpoint only
stores a **path** (`cfg_dir` hparam) + the architecture config — not the resolved
cids.

If those YAMLs are edited after training (re-QC, adding/removing sessions,
reordering), the readout sizes silently drift away from the checkpoint and it
no longer loads. This is a cross-user footgun when `cfg_dir` points into another
user's home directory.

Concrete instance (2026-07): the `resnet_none_convgru` canonical twin
(`multidataset_120_long/.../epoch=147-val_bps=0.5702.ckpt`, trained Dec 2025)
stores `cfg_dir = /home/jake/repos/VisionCore/.../multi_basic_120_long.yaml`.
That YAML was rewritten on 2026-05-14 (20 sessions → 30 sessions, different
cids), so the checkpoint stopped loading for everyone. Declan's run succeeded on
2026-01-30 only because the config still matched then. The fig3 twin
(`digital_twin_120/2026-03-31_.../epoch=374-val_bps=0.6395.ckpt`,
`resnet_concat_convgru`) loads fine because its `cfg_dir` points at the in-repo,
version-controlled config whose cids still match its state_dict.

### Train-time fix

- **Snapshot the resolved dataset configs into the checkpoint** at training time
  (store the actual `cids` per session, not just the `cfg_dir` path). Then the
  checkpoint is genuinely self-contained for readout sizing and is immune to
  later YAML edits.
- **Point `cfg_dir` at the in-repo, version-controlled config**, never at a
  personal home directory (`/home/<user>/...`). The config that produced a
  checkpoint should be pinned alongside it.

### Robust loading (analysis-side, until the train-time fix lands)

Write one shared `load_twin(checkpoint_path)` helper and route all canonical
analyses through it. It should:

1. **Take an explicit, pinned `checkpoint_path`.** Never use
   `scan_checkpoints` + `model_index=0` ("best in dir") for a canonical result —
   it silently repoints as new checkpoints appear or directories change.
2. **Size readouts from the checkpoint's own `state_dict`**, using
   `model.readouts.*.mean` row counts (the pattern already in
   `paper/fig3/_fig3a_data.py:94`), and use the YAML only for cid *identity*.
3. **Validate** that per-session cid counts match the checkpoint counts, and fail
   loudly naming the drifting session ("session Allen_2022-02-16: yaml 116 vs
   ckpt 120") instead of dumping 200 lines of tensor-size mismatches.

---

## 2. `behavior=None` silently skips the concat modulator

### Failure mode

`scripts/spatial_info.py::compute_rate_map` calls
`model.model.core_forward(stim, None)`. In `core_forward`
(`models/modules/models.py:202`) the modulator only runs when
`behavior is not None`:

```python
if self.modulator is not None and behavior is not None:
    feats = self.modulator(feats, behavior)
```

For a **`concat`** twin the recurrent stack is built expecting
`convnet_channels + modulator_dim` channels
(`models/modules/models.py:152-153`). Passing `behavior=None` skips the
concatenation, so `feats` arrives at the recurrent layer with the wrong channel
count — a crash or, worse, silently wrong features. Analysis paths written
against `none`-modulator twins (no behavior) break when repointed at a
behavior-conditioned twin (e.g. the fig3 `resnet_concat_convgru` twin, whose
`behavior_dim = 42`).

Note the 42-dim behavior for the current twin is **entirely eye-movement
derived**: eye velocity through a raised-cosine temporal basis (40 dims) + raw
eye position (2 dims). It is not an extraretinal/arousal signal — so zeroing it
is a real ablation, not a neutral default.

### Robust fix

- A behavior-conditioned twin should **assert** it receives a behavior tensor in
  any inference path, rather than silently no-op'ing the modulator on `None`.
- The shared forward helper should build the correct behavior tensor (from the
  eye trace, matching the training transforms) for concat/FiLM twins, and only
  allow `behavior=None` for genuinely `none`-modulator twins.

---

## 3. Allen/Logan sample imbalance — equalize per-monkey weight

The twin underperforms on Logan's data because training is dominated by Allen:
there are simply far more samples/sessions from Allen than from Logan, so the
shared core is pulled toward Allen's statistics.

Planned fix: **sample evenly from the two monkeys** during training (balanced
per-subject sampling / loss weighting) so Allen and Logan contribute equal
weight, rather than weighting by raw sample count. Revisit on the next training
run.

---

## 4. The (twin, config, mcfarland-outputs) triple must stay mutually consistent

### Failure mode

`scripts/spatial_info.py::get_spatial_readout` builds the population readout by
selecting high-CCNORM units per session:

```python
cids2use = np.where(outputs[...]['ccnorm']['ccnorm'] > .5)[0]
feat_weight = readout.features.weight[cids2use]      # readout has len(cids) units
```

This silently assumes the mcfarland `ccnorm` array is indexed **identically** to
the twin's per-session readout — i.e. the mcfarland outputs were computed on the
exact `cids` the twin was trained on. When they aren't, `cids2use` can index past
the readout (`IndexError: index 35 is out of bounds for dimension 0 with size
35`), or worse, silently map channels to the wrong cells.

Concrete instance (2026-07): repointing the redundancy analysis at the fig3
concat twin (30 sessions, config cids e.g. 116/77/149/…) against jake's
Jan-2026 mcfarland outputs fails — those outputs index the full recorded
population (e.g. 85 units for a session the twin curated to 116, or 15-session
`mono` / 1-session plain variants), so ccnorm indexing ≠ readout cids. There are
three artifacts (`mcfarland_outputs{,_mono,_standard}.pkl`) and **none** align
with the fig3 twin's cids.

The deeper point: a twin checkpoint, its dataset-config `cids`, and any
CCNORM/mcfarland artifact used to build readouts from it are a **coupled triple**.
Regenerating or editing any one without the others breaks the analysis. This is
the same drift class as item 1, one level up.

### Robust fix

- **Version the triple together.** Tie each mcfarland/CCNORM artifact to the
  (checkpoint, config) it was computed against — e.g. store the resolved cids
  and checkpoint hash inside the artifact, and validate them at load.
- **Index CCNORM by cid, not by position.** `get_spatial_readout` should map
  `ccnorm` to the readout via cell identity (cid), and assert lengths match,
  failing loudly with the offending session when they don't — instead of
  positional indexing that can silently misalign.
- **Regenerate mcfarland outputs per twin.** When adopting a new canonical twin,
  regenerate its mcfarland/CCNORM outputs (`mcfarland_sim.run_mcfarland_on_dataset`)
  against that twin + its config, rather than reusing an older artifact.
