# Staged plan: validation -> Tier 1 -> Tier 2

Three stages, run in separate sessions because Tier 1 is a large compute
commitment and must not start until the choices feeding it are settled.

| Stage | What it decides | Reported? | Rough cost |
|---|---|---|---|
| **0. Validation** | Hyperparameters and pipeline improvements. Batch size, learning rate, compute budget, homogeneous vs cross-session batching, subject balancing, readout population. | **No** — these are configuration choices, documented in `MODEL_CARD.md` but not figures. | ~150-200 GPU-h |
| **1. Capacity + defensibility** | Panels A (capacity scaling), C (conclusion stability), F1 (no-modulator lesion), H1 (noise ceiling). Picks and defends the final model. | Yes, supplementary figure | ~570 GPU-h |
| **2. Enrichment** | Panels B (shared vs individual), G (frozen-core transfer), F2 (remaining lesions), H2 (baseline floor). | Conditional on Tier 1 | ~200 GPU-h |

Stage 0 is where we are. Nothing in Stage 0 appears in the paper as a result;
it exists so that Tier 1 runs once, with defensible settings.

---

## The hard constraint: do not break existing figures

Figures 1-4 and the existing supplements depend on the current checkpoint
loading correctly, which depends on `experiments/dataset_configs/sessions/*.yaml`
still resolving to the exact `cids` the checkpoint was trained with. This is
TWIN_IMPROVEMENTS item 1, and it is a live footgun for this project specifically.

**Therefore every Stage 0 change is additive and sandboxed:**

- New session configs go in `paper/model_selection/configs/sessions_v2/`, never
  as edits to `experiments/dataset_configs/sessions/*.yaml`.
- New parent dataset config is a new file, not an edit to
  `multi_basic_120_long.yaml`.
- Changes to shared training code must leave the default code path
  byte-identical. The `samplers.py` fix qualifies: `ByDatasetBatchSampler` is
  only constructed when `homogeneous_batches=True`, a path that previously
  raised `TypeError` on construction and so was unreachable.

**Verification gate before any Stage 0 change is considered done:**
regenerate the fig3 data cache from the *existing* pinned checkpoint and confirm
the reported numbers are unchanged (median CC_norm 0.664, Delta extraretinal
-0.014, Delta stabilized -0.143). If those move, something leaked.

---

## TWIN_IMPROVEMENTS integration

The backlog in `VisionCore/TWIN_IMPROVEMENTS.md` was written for "the next
training cycle." This is that cycle. Mapping:

### Must land before Tier 1 (they change what gets trained)

**Item 5 — twin population must be a superset of fig2 units.** Per-session
readout `cids` must satisfy `fig2_analyzed ⊆ twin_readout`. Currently 5.6% of
fig2 neurons (76 cells) fall below the twin's 200-spike threshold, forcing an
intersection step in `supp_model_replication/`. Fixing it at train time removes
that bookkeeping permanently.
*Sandboxing:* this changes `cids`, which is exactly what breaks checkpoint
loading. It must be a new `sessions_v2/` config set. Do not edit the existing
session YAMLs.

**Item 3 — Allen/Logan sample imbalance.** Balanced per-subject sampling so the
shared core is not pulled toward Allen's statistics. Note this is *coupled* to
the homogeneous-vs-cross-session decision: both determine batch composition, and
subject balancing is implemented differently under each. Decide them together,
not sequentially.

**Item 1 (train-time half) — snapshot resolved `cids` into the checkpoint.**
So the checkpoint is self-contained for readout sizing and immune to later YAML
edits. Must be in place before Tier 1, or every Tier 1 checkpoint inherits the
same fragility.

### Should land before Tier 1 (cheap, prevents silent wrongness)

**Item 2 — `behavior=None` silently skips the concat modulator.** A
behavior-conditioned twin should assert it receives a behavior tensor rather
than no-op'ing the modulator and feeding the recurrent stack the wrong channel
count. Directly relevant: Tier 1 includes a no-modulator lesion (F1), so
`concat` and `none` twins will coexist and be evaluated by shared code.

**Item 1 (analysis-side half) — one shared `load_twin(checkpoint_path)`.**
Size readouts from the checkpoint's own `state_dict`; use YAML for cid identity
only; fail loudly naming the drifting session. Tier 1 produces 7+ checkpoints
that all need loading by the same evaluation code.

### Can wait for Tier 2

**Item 4 — (twin, config, mcfarland-outputs) triple consistency.** Version the
triple together and index CCNORM by cid rather than position. Needed when the
spatial-information analyses are repointed at a new canonical twin.

---

## Stage 0 experiments

All at width 1.0 unless noted, short runs, selection on validation BPS under the
`protocol.py` three-way split. Confirm the winning settings still hold at width
2.0 before locking them in, since the ladder spans 152x and a setting tuned at
one capacity need not transfer.

| ID | Question | Arms |
|---|---|---|
| E1 | Homogeneous vs cross-session batching | `homogeneous_batches` False / True, matched effective batch and sample budget. Report **both** validation BPS and throughput — this is a training-dynamics change, not just an optimization. |
| E2 | Effective batch size | 256 / 1024 / 4096 via accumulation |
| E3 | Learning rate | 3e-4 / 1e-3 / 3e-3, and whether `core_lr_scale` != 1.0 helps |
| E4 | Compute budget | Cosine horizon set so LR actually anneals; find samples-to-convergence. The paper model never annealed (horizon 9999 epochs, reached 374), so this is unmeasured territory. |
| E5 | **Cross-subject interference** (revised) | Allen-only and Logan-only models vs the joint model, on the same sessions. Joint ≈ single ⇒ no interference; joint < single ⇒ negative transfer, and only then is a weighting change warranted. |
| ~~E6~~ | ~~Readout population~~ | **Dropped.** The invariant already held; see below. |

**E5 was rewritten on evidence, 2026-08-04.** It was "subject balancing
off / on". `data_census.py` measured the actual composition — Allen 51.8% of
training samples to Logan's 48.2%, and Logan contributing *more* sessions
(16 vs 14) — so the imbalance the arm was designed to correct does not exist.
Unit counts are lopsided (71/29) but never enter the weighting: the loss is a
masked mean per session, then a mean over sessions in the batch, so per unit
Logan is already weighted ~2.5x more than Allen. `subject_gap.py` further shows
the held-out gap is not distinguishable from zero with sessions as the
resampling unit (CC_norm -0.083, 95% CI [-0.181, +0.041]) and that matching
units on rate and reliability collapses the CC_abs gap to -0.016
[-0.091, +0.058]. Balancing would be tuning against noise. What a census cannot
settle is whether joint training interferes, which is what E5 now measures.

**E6 was dropped.** `fig2_analyzed ⊆ twin_readout` holds by construction — both
populations come from the same session YAML `cids`. The 28 missing cells (2.7%
of fig2's 1022) were lost to fig3's `MIN_TOTAL_SPIKES = 200` *analysis*
threshold, which **stays at 200**: lowering it costs panel D up to half its
scored windows in 8 of 24 sessions, because the base window mask is a
conjunction over every unit in the session. No retraining, and no
`sessions_v2/`.

E1 remains coupled to subject weighting: `ByDatasetBatchSampler` draws a session
with p proportional to its size, so homogeneous batching is the one setting
under which sample-share becomes the effective per-session weight.

---

## Open decisions

1. **Is one-session-per-step batching acceptable in principle?** It changes the
   gradient — each step sees one session rather than all 30. E1 measures the
   cost; whether the dynamics change is acceptable is a judgment call.
2. **Per-run sample budget for Tier 1.** Depends on E4. The measured ladder cost
   assumed sample-matching to the paper model's ~49M samples; a properly
   annealed shorter run may match it, which would cut Tier 1 substantially.
3. **Keep width 0.25?** It is latency-bound and costs ~35 GPU-h for the least
   informative rung. Dropping it saves 6% of Tier 1 but loses a point on the
   left arm of the capacity curve.

---

## Known gaps

Both closed on 2026-08-04:

- **Test suite.** Run with `uv run --with pytest pytest VisionCore/tests/`
  (pytest is still not a workspace dependency; the ephemeral form avoids
  touching the shared `.venv`). Baseline before any Stage 0 change was 8
  failed / 65 passed / 3 errors, all pre-existing and none in the training or
  data path. Reverting the `samplers.py` fix reproduced that result exactly,
  confirming it is non-regressive. After Stage 0 changes: 8 failed / 99 passed
  / 3 errors — same failures, +34 new tests.
- **Gradient checkpointing.** Removed: the unused import in
  `models/modules/convnet.py`, the `checkpointing:` key from all 10 model
  configs, and the three no-op `use_checkpointing` assignments in
  `fig3/_fig3_data.py`, `fig3/_fig3_ablation_data.py`, and
  `supp_model_replication/_supp_inference.py`. Nothing read the attribute, so
  this cannot change behaviour.

New gap found while wiring the sweep:

- **`train_multidataset.py` hard-coded `limit_val_batches=1.0`**, so every
  validation pass ran the full ~5900-batch validation set (~22 min), which
  would have dominated every short run. `--limit_val_batches` and
  `--check_val_every_n_epoch` are now arguments, defaulting to the old
  behaviour; `launch.py` uses 0.1 every 4th epoch.
