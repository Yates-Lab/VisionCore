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
three default to the previous behaviour. Note dataset loading costs ~19 min per
launch, which is ~5 h across the 15 arms on top of training.

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

Not yet built: `collect.py`, `stability.py`, figure scripts, `train_final.sh`.
