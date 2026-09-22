# Model selection and validation

> **Production note:** the retained native-240 model is defined only by
> [`production_model.yaml`](production_model.yaml). The three-stage fitting and
> Figure 3/4 handoff is documented in
> [`../../PRODUCTION_PIPELINE_HANDOFF.md`](../../PRODUCTION_PIPELINE_HANDOFF.md).
> Historical capacity-ladder material below remains useful background but is
> not a selector for the current manuscript model.

Reproducible training, capacity selection, and validation for the digital twin
used in Figures 3 and 4.

## Why this exists

The Figure 3/4 checkpoint was produced by `experiments/train_digital_twin_120_long.sh`
with flags that were not recorded anywhere in `paper/`, and that script's current
defaults no longer reproduce it (its default `MODEL_CONFIG` is the *no-modulator*
variant). The checkpoint itself is pinned by absolute path in `fig3/_fig3_data.py`.
This directory replaces that arrangement with a declared run family, a frozen
evaluation protocol, and a single pinned final model.

It also answers the question a reviewer will ask about any digital-twin result:
whether the conclusions depend on the particular capacity and hyperparameters
that happened to be trained.

## Layout

| File | Role |
|---|---|
| `protocol.py` | Frozen splits, stimulus conditions, inclusion criteria, metrics. Hashed; runs under different hashes cannot be pooled. |
| `gen_configs.py` | Width-scaled model configs for the capacity ladder. |
| `measure.py` | Measured parameter counts, FLOPs, memory, step time. Nothing predicted. |
| `probe_capacity.py` | Runs `measure.py` across the ladder on real batches. Plan the compute budget against its output, not against estimates. |

## Protocol notes

**Three-way split.** The paper model used a two-way 80/20 train/val split, so
checkpoint selection and reported performance drew on the same trials. That is
tolerable for one model but biased for a family: picking the best of N on
validation and then reporting validation is optimistic in N. `protocol.py`
declares 70/15/15.

**Held-out stimulus condition.** The twin is fit only to the three free-viewing
conditions (`backimage`, `gaborium`, `gratings`). The fixated flashed-image
condition (`fixrsvp`) is withheld from fitting entirely and is what every figure
evaluates on. This makes the reported numbers out-of-domain generalization, which
is a stronger claim than a within-distribution test score, and it is now stated
explicitly in the manuscript.

**Batching.** `MultiDatasetDM` defaults to `homogeneous_batches=False`, so one
optimizer step is a *list* of per-session dicts summing to `batch_size`, not a
single tensor batch. Memory and FLOPs must be measured over the whole list.
`probe_capacity.py` does this; anything that measures a single sub-batch will
understate the cost by roughly the session count.

## Capacity ladder

Realized parameter counts (measured, not predicted — channel rounding breaks
exact quadratic scaling):

| width | 0.25 | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|
| total | 0.47M | 1.39M | **4.92M** | 18.46M | 40.63M | 71.43M |
| core | 0.37M | 1.20M | 4.55M | 17.73M | 39.54M | 69.98M |

Width 1.0 reproduces the paper model exactly. The ladder spans 152x and reaches
past the ~80M saturation point reported for mouse visual-cortex foundation
models (Willeke et al. 2026).

## Measured capacity (RTX 6000 Ada, 49 GiB, 30 sessions)

Peak GiB / ms-per-step, measured by `probe_capacity.py` on real batches and real
optimizer steps. Raw data in `capacity_probe.json` and
`capacity_probe_smallbatch.json`.

| width | params | bs64 | bs128 | bs256 | best ms/sample |
|---|---|---|---|---|---|
| 0.25 | 0.47M | — | — | 5.2 / 656 | 2.56 |
| 0.5 | 1.39M | — | — | 8.5 / 412 | 1.61 |
| 1.0 | 4.92M | — | — | 16.1 / 524 | 2.05 |
| 2.0 | 18.5M | 8.4 / 380 | 16.2 / 566 | 31.7 / 1027 | 4.01 |
| 3.0 | 40.6M | 13.0 / 597 | 24.5 / 1040 | OOM | 8.12 |
| 4.0 | 71.4M | 17.8 / 849 | 33.1 / 1479 | OOM | 11.56 |

**Operating points: bs=256 for width <= 2, bs=128 for width 3 and 4**, with
gradient accumulation set to hold the effective batch at 1024. Larger
micro-batches are more efficient per sample at every width, so use the largest
that fits.

Memory grows *linearly* with width, not quadratically, because it is
activation-dominated (activations scale with channel count, parameters with its
square). This is why 71M parameters fits in 18 GiB at bs=64.

### Two findings from the probe

**Gradient checkpointing is not implemented.** `torch.utils.checkpoint.checkpoint`
is imported at `models/modules/convnet.py:4` and never called; `use_checkpointing`
appears nowhere in `models/`. The `checkpointing:` key in every model config is
inert, and `model.model.convnet.use_checkpointing = False` in `fig3/_fig3_data.py`
and `fig3/_fig3_ablation_data.py` sets an attribute nothing reads. Harmless in the
figures, but there is no memory escape hatch. It is not needed: reducing the
micro-batch covers the whole ladder.

**Training is launch-latency-bound at the small end.** With
`homogeneous_batches=False` (the default, and what the training script uses), one
optimizer step is ~30 sequential sub-forwards of ~8 samples each. Width 0.25 is
*slower* per step than width 1.0 despite 10x fewer parameters. Consequences: small
rungs are not cheap in wall-clock, and FLOPs rather than wall-clock is the correct
x-axis for the capacity-scaling panel.

## Naming

`E1a`, `F3c` and "Stage 0b" carry no meaning, so reading a result table meant
holding `launch.py`'s arm definitions in your head. From experiment 02 onward:

- **Experiments** are `NN-slug` — `02-lr-batch-landscape` — declared in
  `experiments.py` with their baseline. Notes live in `notes/NN-slug.md`.
- **Runs** are `NN_<label>` — `02_lr3e-3_bs128` — where the label names every
  knob the experiment varies, at that run's value.

Two rules make the label derivable rather than declared:

**Axes are detected, not listed.** An experiment's axes are the knobs that
actually vary across its arms, minus any knob the others determine. Experiment
02 varies `batch_size`, and effective batch, accumulation and epoch count all
follow from it, so only `bs` appears; experiment 01 varies `effective_batch`
directly with the micro-batch fixed, so `eb` appears there. Neither case is
hard-coded — `axes_for` decides, the way `config_signature` detects replicate
groups instead of restating them.

**A label is complete over the axes, not over the differences.** Every axis
appears in every label including at its baseline value: `02_lr3e-3_bs256`, not
`02_lr3e-3`. Naming only the differing knobs makes a label relative to the
baseline, so it changes meaning if the baseline moves.

Runs from experiments 00 and 01 keep their `E`/`F` directory names — the ids
are woven through the notes, the frozen arm definitions and the history, and
renaming the frozen record buys readability in runs nobody will launch again.
They get a *derived* label instead, so `collect.py` prints `F2a` as
`lr1e-3_eb256_noadapter` with no directory renamed and no manifest rewritten.

## Staging

See `VALIDATION_PLAN.md`. Three stages in separate sessions:
**Stage 0 (validation, current)** settles unreported hyperparameters and folds in
the `TWIN_IMPROVEMENTS.md` backlog; **Tier 1** runs the capacity ladder and the
panels that defend figures 3 and 4; **Tier 2** is conditional enrichment.

Nothing in Stage 0 is reported. It exists so Tier 1, which is ~570 GPU-h, runs
once with defensible settings.

## Status

Built: `protocol.py`, `gen_configs.py`, `measure.py`, `probe_capacity.py`,
`VALIDATION_PLAN.md`, `data_census.py`, `subject_gap.py`, `launch.py`,
`MODEL_CARD.md`, `configs/multi_120_long_split3*.yaml`.

`MODEL_CARD.md` is the entry point for what has been settled and why.

| File | Role |
|---|---|
| `data_census.py` | Per-session sample/unit/spike census, and the data-path regression gate: hashes every served split index so a shared-code change can be proven inert. `--diff before.json after.json`. |
| `subject_gap.py` | Whether the Allen/Logan held-out gap survives matching on unit difficulty. Coarsened exact matching, session-clustered bootstrap. |
| `launch.py` | The Stage 0 run family. Sample-budgeted arms, protocol-stamped manifests. `--list` to see them. |
| `regen_fig3_caches.py` | Rebuilds the fig3 inference caches after the spike-threshold change. |
| `collect.py` | Pools the arms into one table. Refuses to mix protocol hashes, skips unmanifested directories by name, reports an unevaluated run as absent rather than zero. `--json`. |
| `stability.py` | The ΔBPS floor from the baseline replicates, and a resolved / marginal / unresolved verdict per arm. |
| `train_final.sh` | Trains the single pinned final model. Contains no training flags of its own; delegates to `launch.py FINAL`, which refuses to build until `FINAL_SETTINGS` is filled in. |

Two premises from the original plan did not survive measurement: the twin
population already covered fig2 (so no `sessions_v2/`, no retraining), and the
Allen/Logan sample imbalance does not exist (so no subject balancing). Both are
documented with evidence in `MODEL_CARD.md` and `TWIN_IMPROVEMENTS.md`.

Fixed in VisionCore: `training/samplers.py:271` — `ByDatasetBatchSampler` passed
a `data_source` to `Sampler.__init__`, which torch >= 2.6 no longer accepts, so
`homogeneous_batches=True` raised `TypeError` on construction and was
unreachable. Now verified to yield full single-session batches. The default
(`homogeneous_batches=False`) path never constructs this sampler and is
unchanged.

Fixed in VisionCore, 2026-08-05: the same sampler **yielded the identical batch
sequence every epoch**. It seeded on `self.seed + self._step`, and `_step` moves
only via `set_step`, called only by `CurriculumCallback`, registered only under
`--enable_curriculum` — which no arm passes. With `limit_train_batches=512` a run
therefore trained on one epoch's 131,072 samples (~1.6% of the training set)
repeated for its full duration. Found by watching arm E1b's validation BPS
flatline; E1b was killed and voided. The seed now includes a per-pass epoch
counter, leaving the first pass unchanged so seeded re-runs still reproduce.
Regression tests: `tests/test_by_dataset_batch_sampler.py`, whose three
defect-targeting cases were confirmed to fail against the unfixed sampler.

Blast radius is one run. `_mk_loader` takes the distributed branch before the
homogeneous one, so the DDP-trained paper model never touched this sampler, and
the sampler was unreachable at all before the `data_source` fix above. The
defect could only ever have reached a single-GPU homogeneous run, of which E1b
was the first and only.

Three-way split wiring is done: a `test_split` key switches `prepare_data` to
the three-way splitter and `MultiDatasetDM` exposes `test_dataloader()`; absent
the key the two-way path is byte-identical, verified by unit tests and by a
real-data before/after hash of every served index set.

Training pipeline validated end to end on 2026-08-04, by running it rather than
by inspection: checkpoints save; the cid snapshot round-trips through a saved
file (`load_twin` reports `cids_source == 'checkpoint'`, 30 sessions / 2790
units, validated against the state_dict's own readout sizes); homogeneous
batching trains real steps for the first time; `--ckpt_path` resume restores and
continues. `train_multidataset.py` gained `--ckpt_path` (resume was absent, not
merely unverified), `--seed`, and `--early_stopping/--no-early_stopping`; all
three default to the previous behaviour. Dataset loading costs ~3 min per launch
(measured 2026-08-05, warm page cache; an earlier ~19 min figure was ~6x too
high). Nothing is shared between runs and evaluation pays it twice -- see
`STAGE0B_NOTES.md`.

`evaluate.py` pins the test-split pass to **cross-session batching for every
arm** (`EVAL_HOMOGENEOUS_BATCHES = False`), whatever the arm was trained with.
It previously inherited `homogeneous` from the run manifest, which would have
scored a homogeneous arm on ~63% of the test split — `ByDatasetBatchSampler`
draws with replacement across batches — while a cross-session arm was scored on
100%, and the two differenced as if commensurable. Batch composition affects
gradients, not a forward-only metric, so pinning it changes no arm's training.
E1a's reported 0.6090 is unchanged, having trained cross-session already.

`evaluate.py` scores a run on the metrics `protocol.py` declares. Two passes:
the three-way **test split** on the fit conditions (built by the split wiring
and, until now, never scored — nothing in the repo called `test_dataloader()`),
and the held-out **`fixrsvp`** condition for CC_norm and single-trial r², the
number Figures 3 and 4 rest on. It scores the test split with the same
`PoissonBPSAggregator` the training loop uses rather than adding a `test_step`
to `MultiDatasetModel`, so no shared training code changes. `psth_r2` is read
from Figure 2's aligned cache rather than recomputed, so inclusion selects the
same neurons Figure 2 reports on. Unit-tested in
`tests/test_model_selection_evaluate.py`; the two GPU passes are pending their
first end-to-end run.

The collection layer is built (2026-08-04). `collect.py` pools the runs and
`stability.py` gives their deltas a scale, from the baseline replicate group it
finds by *configuration signature* rather than by a hard-coded name list — that
E1a, E2b and E3b are the same configuration is a consequence of `launch.py`'s
defaults, and a consequence is a thing to detect, not to restate somewhere it
can fall out of date. Neither is usable as a decision aid yet: with only E1a run
there is one replicate, so `stability.py` reports no floor and names E2b and E3b
as what is missing. Both are unit-tested in `tests/test_model_selection_collect.py`
(21 tests) and both have been run against the real checkpoint root.

`train_final.sh` carries no training flags. It delegates to `launch.py FINAL`,
whose `FINAL_SETTINGS` is a dict of `None` placeholders that the sweep fills in;
`resolve` refuses to build a command while any remain, and a test asserts the
final model's flag set is identical to an arm's. The indirection is deliberate:
`experiments/train_digital_twin_120_long.sh` kept its own copy of the flags,
drifted from the checkpoint it supposedly produced, and left the paper model's
real settings recorded nowhere — which is the reason this directory exists. A
second shell script with a second copy of the flag list would rebuild that
defect.

Not yet built: figure scripts.
