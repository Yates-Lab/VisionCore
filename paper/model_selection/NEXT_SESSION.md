# Next session — Stage 0: run E1b, then the rest of the sweep

Copy the block below into a fresh session.

---

We are in **Stage 0 (validation)** of `VisionCore/paper/model_selection/`, the
reproducible training and model-selection pipeline for the digital twin behind
Figures 3 and 4. Read `paper/model_selection/MODEL_CARD.md` first — it records
every choice already settled and the evidence for it — then `VALIDATION_PLAN.md`
and `README.md`. `VisionCore/TWIN_IMPROVEMENTS.md` has a status table at the top.

**The training and evaluation pipeline is now validated end to end, and the
first arm (E1a) has run.** What remains is running the other 14 arms and
building the collection layer.

## Hard constraints

1. **One GPU at a time for training.** The box has 2x RTX 6000 Ada and the
   second must stay usable by other users. Pass `--gpu N` to `launch.py` and run
   arms **serially**. Short analysis jobs (minutes — e.g. an evaluation
   shakedown) on the other card are fine; the constraint is about not walling a
   GPU for days, not about never touching it. Check `nvidia-smi` before
   launching.
2. **Do not break Figures 1-4.** The regression gate is
   `paper/model_selection/data_census.py`: it hashes every served split index
   per session. Run it before and after any shared-code change and diff:
   ```
   uv run python paper/model_selection/data_census.py --out after.json
   uv run python paper/model_selection/data_census.py --diff paper/model_selection/census_baseline.json after.json
   ```
   `census_baseline.json` (all 30 sessions) is the standing reference. Expect
   `Data path unchanged across 30 sessions`.
3. **Never edit** `experiments/dataset_configs/sessions/*.yaml` or
   `multi_basic_120_long.yaml`. Model-selection configs live in
   `paper/model_selection/configs/` and point back at the unmodified session
   YAMLs, so the trained population is identical to the figure population.
4. `pytest` is not a workspace dependency; run it as
   `uv run --with pytest pytest VisionCore/tests/`. Baseline is **8 failed /
   107 passed / 3 errors** — all 8 failures and 3 errors are pre-existing and
   outside the training and data paths. Treat any *new* failure as a regression.

## The work, in order

### 1. E1b — the arm that answers the question E1 exists for

```
uv run python paper/model_selection/launch.py E1b --gpu 0
```

E1a (done) is the **control**: cross-session batching, the current default and
what the paper model was trained under. E1b is homogeneous batching — one
session per optimizer step. This is the only arm that tests H1, and it is doubly
loaded, which is why E1 cannot be split into two experiments:

- **Gradient.** Under E1a the shared core sees a gradient averaged over ~30
  sessions per step; under E1b it sees one session per step.
- **Subject weighting.** `ByDatasetBatchSampler` draws a session with p
  proportional to its size, so E1b makes *sample* share the effective
  per-session weight. E1a weights each session in the batch equally, which per
  unit already favours Logan ~2.5x over Allen. E1 is therefore the
  subject-weighting experiment, and the only one — the census refuted the
  imbalance premise that motivated a separate arm.

E1b ≈ E1a on validation BPS ⇒ adopt homogeneous batching for its throughput,
which matters against Tier 1's ~570 GPU-h. E1b < E1a ⇒ cross-session stays, and
sample-proportional weighting is shown to hurt.

**E1b was run on 2026-08-05, found void, and must be run again.** Its
validation BPS flatlined (epoch 7 → 11, 0.3508 → 0.3509) against E1a's
0.418 → 0.465, which exposed a defect in `ByDatasetBatchSampler`:

- `__iter__` seeded on `self.seed + self._step` (`training/samplers.py`);
- `_step` advances only through `set_step`, whose sole caller is
  `CurriculumCallback` (`training/callbacks.py:282`);
- that callback is registered only under `--enable_curriculum`
  (`train_multidataset.py:266`), which no arm passes.

So `_step` stayed 0 and **every epoch yielded the identical batch sequence**.
With `limit_train_batches=512`, E1b trained on the same 131,072 samples (~1.6%
of the 8.16M-sample training set) 61 times over. It measured nothing about
homogeneous batching, in either direction. The run was moved to
`<CKPT_ROOT>/../model_selection_void/E1b_epoch_repeat_2026-08-05/`.

Fixed by folding a per-pass epoch counter into the seed; the first pass is
unchanged, so seeded re-runs still reproduce. Regression tests in
`tests/test_by_dataset_batch_sampler.py` (9 tests); the three that target the
defect were confirmed to fail against the unfixed sampler.

**Nothing else is affected.** `_mk_loader` takes the distributed branch before
the homogeneous one, so the DDP-trained paper model never used this sampler,
and before commit `131292f` the sampler raised `TypeError` on construction and
was unreachable. E1a used `--no-homogeneous_batches`, a plain shuffling
DataLoader that reshuffles each epoch. No figure and no prior result is touched.

**The lesson for the rest of the sweep:** the previous handoff recorded
homogeneous batching as "verified to *train* (2-epoch smoke)". It does train —
identical epochs are invisible unless batch indices are compared *across*
epochs. "It ran" is not "it is doing the right thing", and a flat metric is a
symptom worth chasing before it is worth interpreting.

### 2. The remaining arms

`launch.py --list` shows all 15. After E1b: E2a, E2c, E3a, E3c, E3d, E3e, E4a,
E4b, E5a, E5b, C1, plus the two baseline replicates E2b and E3b.

**E4b (16M samples) is now the most informative arm.** E1a's validation BPS was
still climbing +0.007 per validation at the end of its anneal, so the 8M budget
sits below the knee. E4 governs Tier 1's per-run budget, the largest single
lever on its ~570 GPU-h.

### 3. Evaluate each run

```
uv run python paper/model_selection/evaluate.py <RUN> --gpu 0
```

~65 min per run. Two passes: the three-way test split (BPS on trials selection
never saw) and the held-out `fixrsvp` condition (CC_norm, single-trial r², BPS —
out-of-domain generalisation, and what Figures 3 and 4 rest on). Writes
`evaluation.json` beside the run's manifest. Unit-tested in
`tests/test_model_selection_evaluate.py`; both GPU passes have been run for real.

`--max-datasets N`, `--sessions ...`, `--skip-test-split`, `--skip-fixrsvp` make
a capped shakedown possible in minutes. A capped run is a shakedown, never a
reported number.

#### 4. Build the collection layer — DONE (2026-08-04)

`collect.py`, `stability.py` and `train_final.sh` are built, unit-tested
(`tests/test_model_selection_collect.py`, 21 tests) and run against the real
checkpoint root. See `README.md` for what each does. Two things to know:

- `stability.py` reports **no floor** until E2b and E3b run — with one
  replicate there is no spread, and it names the two arms that are missing
  rather than printing a number that would be read as one.
- `train_final.sh` holds no training flags; `launch.py FINAL` builds them with
  the same `build_command` as the arms and refuses while `FINAL_SETTINGS` still
  has `None` placeholders. Fill those in from the sweep, not by hand.

Still to do here: finish `MODEL_CARD.md`, whose hyperparameter sections stay
open until E2/E3/E4 report.

### The box is shared — check for *sustained* idle, not a snapshot

On 2026-08-04 both cards were occupied for hours by another user's inference
shards. A single `nvidia-smi` is not enough to launch a multi-hour arm against:
one sample showed GPU 1 at 0% util / 571 MiB while that job was merely between
shards. Require a card to be clean — no compute process **and** util ≤ 5% — on
every sample across ~10 minutes before launching, and reset the streak on any
dirty sample.

## Settled — do not redo, do not relitigate

- **`MIN_TOTAL_SPIKES` stays at 200**, in both `paper/fig3/_fig3_data.py` and
  `protocol.py`. Standing user directive. It was lowered to 0 on 2026-08-04 and
  reverted the same day; caches and figures were restored, and Figure 3
  reproduces its published numbers exactly (Panel C N=984, intact 0.664,
  Δ extraretinal −0.014, Δ stabilized −0.143; Panel D N=969, median 1048 scored
  windows/unit, matched subset 505). The 0-spike caches are kept as
  `outputs/cache/*.spikethresh0.bak`.
- **Protocol hash `78ea1cc91f00` is frozen.** Changing `protocol.py` invalidates
  every run stamped under it.
- **No early stopping** for Stage 0 arms (`--no-early_stopping`, set by
  `launch.py`). Arms are sample-budgeted and the cosine horizon *is*
  `max_epochs`; stopping early would leave the LR un-annealed and the arms not
  compute-matched. Note the semantics that made the old default a tripwire:
  Lightning counts patience in *validation calls*, so `patience=10` with
  `check_val_every_n_epoch=4` was 40 of a 61-epoch run.
- **E1a, E2b and E3b are the same configuration**, kept deliberately as baseline
  replicates R1/R2/R3 at seeds 101/102/103. Every other arm reports a ΔBPS and
  needs a run-to-run spread to be read against. Caveat: `split_inds_by_trial*`
  re-seeds globally to `SPLIT_SEED`, so replicates differ in weight init and GPU
  nondeterminism but **not** in data order — the floor understates true spread,
  and a delta near it is unresolved, not null.
- **The shared-window-mask conflation is latent, not active. Do not "fix" it and
  do not recompute Figures 2 or 3 for it.** `VisionCore/covariance.py:1153` and
  `paper/fig3/_fig3_explainable_variance.py:98` gate a (trial, bin) on *every*
  cell being finite, conflating per-cell data availability with the shared
  eye-position criterion. Measured directly across all 25 aligned sessions:
  every cell's NaN pattern is identical (NaN marks an unreached assembly slot,
  never a missing cell), and eye-only vs current masks both yield **117,727
  windows — 0.0% difference**. It would matter only if a future dataset
  introduced genuinely per-cell gaps, where `extract_valid_segments`'s 36-bin
  floor amplifies hard (3.3% of bins → 55% of windows). Investigated and closed
  2026-08-04.
- **Items 1, 2, 3 and 5 of `TWIN_IMPROVEMENTS.md`** are closed; see its status
  table. Item 4 is deferred to Tier 2.

## Verified by running, 2026-08-04 — do not re-verify

Every item on the previous session's "unverified" list was exercised for real:

- checkpoints save (`ModelCheckpoint`, `save_top_k=3` + `last.ckpt`);
- **the cid snapshot round-trips through a saved file** — `load_twin` reports
  `cids_source == "checkpoint"`, 30 sessions / 2790 units, validated against the
  checkpoint's own `state_dict` readout sizes;
- homogeneous batching trains real steps (`ByDatasetBatchSampler`);
- `--ckpt_path` resume restores and continues.

Corrections to the previous handoff:

- `--ckpt_path` did not exist at all — it was **absent**, not merely unverified.
  It exists now (`train_multidataset.py`), and `launch.py --resume` passes it.
- wandb metric logging was already unconditional; `--enable_logging` only adds
  the expensive `ModelLoggingCallback`. Leave it off.
- `launch.py --list` shows **15** arms, not 16.

New in `train_multidataset.py`, all defaulting to previous behaviour:
`--ckpt_path`, `--seed`, `--early_stopping/--no-early_stopping`.

Two defects that only surfaced by *running* `evaluate.py`, both now fixed and
worth knowing about if you write another evaluation path: the datasets are
stored `bfloat16` while weights are float32, so a hand-rolled inference loop
needs `torch.autocast` (training only survives it via `precision="bf16-mixed"`);
and `bits_per_spike` sanitises NaN in the rates but **not** in `dfs`, so
NaN-padded assembled trials silently produce an all-NaN metric rather than an
error.

## E1a result

61/61 epochs, 5 h 54 min, exit 0, no OOM. Best checkpoint
`epoch=59-val_bps_overall=0.5958.ckpt`.

Validation BPS climbed monotonically and never plateaued: 0.295, 0.418, 0.465,
0.513, 0.524, 0.535, 0.548, 0.562, 0.572, 0.577, 0.582, 0.589, **0.596**. The
paper model reached 0.6395 at epoch 374 (~49M samples) with a cosine horizon of
9999 epochs, so its LR never annealed; E1a recovers ~93% of that at ~1/6 the
samples, under a schedule that does anneal. Not a like-for-like comparison —
different split (70/15/15 vs 80/20) and a 10% validation subset — so read it as
indicative and let E4 settle samples-to-convergence.

### E1a evaluation (`evaluate.py`, 2026-08-04)

Written to `<CKPT_ROOT>/E1a/evaluation.json`. Scored from
`epoch=59-val_bps_overall=0.5958.ckpt`, `cids_source == "checkpoint"`.

| | |
|---|---|
| Test-split BPS (30 sessions) | **0.6090** (per-session 0.308–1.058, median 0.635) |
| `fixrsvp` median CC_norm | **0.589** (889 units, 17 sessions) |
| `fixrsvp` median single-trial r² | **0.0303** (892 units) |
| `fixrsvp` median BPS | 0.132 |

Two things worth carrying forward:

- **Test BPS (0.609) came out above validation BPS (0.596).** Not a paradox and
  not evidence of anything: training validated on `limit_val_batches=0.1`, a
  fixed 10% subsample, while the test pass scored 100% of the test split. The
  two numbers are not measured the same way, so do not read the difference as
  generalisation.
- **The held-out CC_norm gap is the informative number.** Figure 3 reports the
  paper twin at median CC_norm **0.664**; E1a reaches **0.589**. The populations
  are close but not identical (fig3 gates on fig2's `cd_population` — rate > 2 Hz
  and PSTH R² > 0.10 — over 984 cells in 19 sessions; `evaluate.py` applies
  `protocol.py` inclusion, 892 units in 17 sessions), so this is suggestive
  rather than a like-for-like contrast. But it points the same way as the
  validation BPS gap: **the 8M-sample budget costs real out-of-domain
  generalisation, not just training-objective BPS.** That is the strongest
  argument yet for E4b, and for not locking Tier 1's per-run budget at 8M.

**Corrected 2026-08-05 — the optimisation proposed here was backwards.** The
previous note observed that evaluation inherits `homogeneous` from the run
manifest, so a cross-session arm scores its test split as ~180,600 eight-sample
sub-forwards instead of 6,020 full 256-sample batches, and suggested forcing
`homogeneous_batches=True` for evaluation as "safe and several-fold faster".

The speed claim was right; the safety claim was wrong. `ByDatasetBatchSampler`
picks each batch's session with `multinomial(..., replacement=True)` and redraws
within the session per batch, so **iterating it covers only ~63% of the unique
samples and scores some of them twice**. Forcing it on would have turned every
arm's test BPS into a random subsample of the split — including E1a's.

`evaluate.py` now pins evaluation to cross-session batching for every arm
(`EVAL_HOMOGENEOUS_BATCHES = False`, `build_test_datamodule`), so all arms are
scored over the same, complete data whatever they were trained with. The sampler
is a training device, not a scoring device.

E1a's existing `evaluation.json` is unaffected — it trained with
`homogeneous=False`, so the inherited value already equalled the pinned one and
`test_split.bps_overall = 0.6090` still stands. Homogeneous arms (E1b) would
have been mis-scored had this not been caught, since their manifests say `true`.

## Budget, measured rather than estimated

- E1a took **5 h 54 min**, not the estimated 4.6 h. The remaining 14 arms are
  therefore ~**85-90 h** serial on one GPU, not ~74 h.
- **Dataset loading costs ~3 min per launch** (measured 2026-08-05 on F1a:
  3 min 05 s from process start to first training step, all 30 sessions). The
  ~19 min previously recorded here was ~6x too high. It is paid again by every
  evaluation -- twice, in fact; see `STAGE0B_NOTES.md`.
- A full `evaluate.py` run is ~65 min (~40 min test split, ~25 min fixrsvp).
