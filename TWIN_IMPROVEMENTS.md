# Twin Loading & Training — Improvement Backlog

Running list of digital-twin fragilities and planned improvements. We append
line items here as we hit them, then do a consolidated refactor + retraining
run once there's time for another training cycle and polish.

**Status as of 2026-08-04** (Stage 0 of `paper/model_selection/`):

| Item | Status |
|---|---|
| 1. Readout-size drift | **Done.** `eval/load_twin.py` sizes readouts from the checkpoint's `state_dict` and names the drifting session; `MultiDatasetModel` now snapshots resolved cids into `hparams.dataset_cids`, so new checkpoints are self-contained. |
| 2. `behavior=None` skips the modulator | **Done.** `models/modules/models.py::require_behavior` raises in both `core_forward` implementations. |
| 3. Allen/Logan imbalance | **Resolved — premise refuted.** See below; do not implement balancing on this basis. |
| 4. (twin, config, mcfarland) triple | Open, deferred to Tier 2. |
| 5. fig2-superset population | **Resolved — invariant already held.** Fixed analysis-side instead; see below. |

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

> **RESOLVED 2026-08-04 — the stated mechanism does not exist. Do not implement
> per-subject balancing on this basis.** Evidence:
> `paper/model_selection/data_census.py`, `paper/model_selection/subject_gap.py`.

The original claim was that "there are simply far more samples/sessions from
Allen than from Logan". Measured over all 30 training sessions:

| subject | sessions | units | train samples | session share | sample share | unit share |
|---|---|---|---|---|---|---|
| Allen | 14 | 1989 | 4,227,251 | 46.7% | **51.8%** | 71.3% |
| Logan | 16 | 801 | 3,935,261 | 53.3% | **48.2%** | 28.7% |

Sample counts are near-equal and Logan contributes *more* sessions. Only unit
counts are lopsided, and unit count never enters the weighting: `MaskedLoss` is
a masked mean over (samples x units) within a session, and `training_step`
averages over the sessions present in the batch
(`multidataset_model.py:473`). With `homogeneous_batches=False` (the default,
and what trained the paper model) a 256-sample batch over 30 sessions contains
essentially every session, so **each session carries equal weight per step**
regardless of its size or unit count. Per unit, Logan is therefore already
weighted ~2.5x more heavily than Allen. Equalizing subjects would slightly
*reduce* Logan's weight.

The held-out difference is real but not reliably nonzero: median CC_norm 0.653
(Allen, 1255 units) vs 0.570 (Logan, 448), a gap of -0.083 whose
session-clustered 95% CI is [-0.181, +0.041]. Logan's session medians span
0.398-0.860, so between-session variance dominates. Logan's included cells are
also harder: 10.3 vs 19.3 Hz median rate, split-half PSTH R^2 0.066 vs 0.178.
Matching units on rate and reliability shrinks the CC_abs gap from -0.131
[-0.214, -0.020] to **-0.016 [-0.091, +0.058]** — i.e. essentially all of the
absolute deficit is population difficulty, not fit quality.

One mechanism remains untested: whether joint training *interferes* with either
subject. That is a training question a census cannot answer, so it is now
Stage 0 arm E5 (`launch.py`), comparing Allen-only and Logan-only models
against the joint model on the same sessions. A weighting change is warranted
only if that shows negative transfer.

Note the coupling: `ByDatasetBatchSampler` (i.e. `homogeneous_batches=True`)
draws a session with p proportional to its size, which *would* make sample
share the effective weight. Arm E1 therefore doubles as the subject-weighting
manipulation.

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

---

## 5. Standardize the twin's population to be a superset of the fig2 units

> **RESOLVED 2026-08-04 — the invariant already held; no retraining needed.**
>
> `fig2_analyzed ⊆ twin_readout` is true *by construction*. Both populations are
> built from the same per-session YAML `cids`: fig2's `_align_one_session`
> (`paper/covariance_decomposition/data_loading.py:95`) calls `prepare_data` on
> the session config, so its `neuron_mask` indexes cids-space, and it aligns
> with `min_total_spikes=0`. The twin's readout is `len(cids)`. fig2 cannot
> include a unit the twin has no readout for.
>
> The cells that went missing were lost to an *analysis* threshold, not a
> training population: `paper/fig3/_fig3_data.py` filtered inference units at
> `MIN_TOTAL_SPIKES = 200`. Under the current inclusion rules (rate > 2 Hz,
> PSTH R^2 > 0.10, 10-unit session floor) that dropped 28 of fig2's 1022
> analyzed cells (2.7%) — the figure quoted below (76 cells, 5.6%) predates the
> R^2 > 0.10 threshold.
>
> The invariant is now asserted at load time by `eval/load_twin.py`.
>
> `MIN_TOTAL_SPIKES` in `_fig3_data.py` **stays at 200**. Lowering it to 0 was
> tried on 2026-08-04 and reverted the same day: panel D scores every cell in a
> session on one shared window set (`np.isfinite(robs).all(axis=2)`,
> `_fig3_explainable_variance.py:98`), so admitting a cell that was isolated for
> only part of a session deletes its absent bins for all of its neighbours. Ten
> of 24 sessions lost base windows, 8 by more than a quarter, to recover 25
> panel C cells. fig3 therefore reports its published numbers unchanged. See
> `paper/model_selection/MODEL_CARD.md`.

### Motivation

The fig3→fig2 replication supplement (`paper/supp_model_replication/`) has to
compare the twin against fig2's covariance-decomposition population, and the two
populations are defined by different inclusion rules that don't nest:

- **fig2** (`covariance_decomposition`): rate > 2 Hz **and** split-half PSTH
  R² > 0.05, then a ≥10-analyzed-unit session floor.
- **twin** (`fig3/_fig3_data.py` inference): > 200 spikes on the fixrsvp
  inference trials (`MIN_TOTAL_SPIKES`).

### Extent of the mismatch (quantified 2026-07, current fig3 twin)

Sessions align cleanly: all 23 fig2-reported sessions are covered by the twin's
24 inference sessions (the one extra, `Logan_2019-12-26`, is a session fig2 drops
at its floor). The mismatch is at the neuron level, within shared sessions:

- **94.4%** of fig2's analyzed neurons (1279 / 1355) have a twin match.
- **76 fig2 neurons (5.6%)** fall below the twin's 200-spike threshold — a
  marginal, low-reliability tail (median rate 3.3 vs 18.7 Hz, PSTH R² 0.095 vs
  0.203). Worst per-session loss is `Logan_2019-12-30` (12 / 24). Logan recovers
  87.8% vs Allen's 96.2%.
- **408 twin neurons (24.2% of the twin's 1687)** are NOT in fig2's analyzed set
  — cells that clear 200 spikes but fail fig2's rate/PSTH-R² inclusion (i.e.
  unreliable cells fig2 deliberately excludes).

The supplement currently works around this by analyzing the **per-session
intersection** (the 1279 both-included cells), so both axes describe the same
population. This is fine but loses 5.6% of fig2's cells and requires an
intersection step in every panel.

### Train-time fix (next run)

- **Train the twin on a population that is a superset of the fig2 analyzed
  units.** Concretely, ensure the per-session readout `cids` include every unit
  that passes fig2's inclusion (rate > 2 Hz & PSTH R² > 0.05), so the twin can
  predict every fig2 cell. Then the replication supplement reduces to "apply
  fig2 inclusion," with no cells lost and no intersection bookkeeping.
- Keeping the twin's superset broader than fig2 (extra low-yield cells) is fine —
  the supplement just re-applies fig2's inclusion on top. The invariant that
  matters is **fig2_analyzed ⊆ twin_readout**, per session.
