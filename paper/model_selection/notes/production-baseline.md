# The Figure 3/4 production model, scored on the sweep's terms

**Why this note exists.** The sweep kept asking "is this good?" against its own
replicate floor, with no external reference. The production twin — the
checkpoint Figures 3 and 4 actually use — is that reference, and until
2026-08-11 it had never been scored under `evaluate.py`. Raw report:
[`../production_baseline.json`](../production_baseline.json).

## What the production model is

From `paper/fig3/_fig3_data.py:52-57`:

```
/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120/
  2026-03-31_11-33-32_learned_resnet_concat_convgru_gaussian/
  learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4/
  epoch=374-val_bps_overall=0.6395.ckpt
```

Width 1.0, adapter **on**, batch 256 × accumulate 4 (effective 1024), lr 1e-3,
wd 1e-5, cross-session batching, ~49M samples, cosine horizon 9999 so the lr
**never annealed**, and `multi_basic_120_long.yaml` with `train_val_split: 0.8`.

**No sweep run is equivalent to it.** `E1a` is the closest — `_defaults` was
built to reproduce its hyperparameters exactly — and still differs in three
ways: 8M samples against 49M, an annealing cosine horizon, and 70/15/15 against
80/20. Everything after `E1a` moved further away (adapter off, homogeneous
batching, batch 128, width 0.5→2.0).

## Why only one metric is comparable

**Validation and test BPS are not.** Production's 0.6395 was computed on
different trials, by a model trained on 80% of trials where sweep arms get 70%.
Against `05_lr5e-4`'s 0.6222 the comparison is invalid in both directions — it
shows neither improvement nor regression.

**`fixrsvp` is.** It is withheld from fitting entirely
(`protocol.HELD_OUT_TYPES`) for the production model and every sweep arm alike,
so the evaluation trials are the same set regardless of how the free-viewing
trials were split. This is what makes the row below meaningful.

**The population matched exactly**: 17 sessions, 892 units, n=889 for CC_norm —
identical to all 17 scored sweep runs. This was the risk that could have voided
the comparison (the production readouts were built under
`multi_basic_120_long.yaml`, the sweep uses `multi_120_long_split3.yaml`) and it
did not materialise.

## Result

| metric (`fixrsvp`, held out) | production | `05_lr5e-4` | Δ | vs floor |
|---|---|---|---|---|
| single-trial r² | **0.0416** | 0.0347 | +0.0069 | **1.58× (resolved)** |
| CC_norm | 0.6688 | 0.6576 | +0.0112 | 0.13× (unresolved) |
| BPS | 0.1719 | 0.1408 | +0.0312 | no floor established |

**The sweep has not yet matched the production model**, and the gap is on
single-trial r² — the quantity Figures 3 and 4 rest on. CC_norm is a tie at
0.13× its floor, and is not a selection metric in any case.

## What the gap does *not* establish

It does not show the sweep's configuration is worse. Production had two
advantages the sweep gave up deliberately:

- **80% of trials against 70%.** The three-way split costs ~14% of the training
  data by construction, to stop selection and reporting drawing on the same
  trials. That cost is the price of an unbiased selection procedure and is not
  recoverable within the sweep — but it *is* recoverable for the deliverable:
  the final model will be retrained on train + val (85%), see
  [final-model-training-split.md](final-model-training-split.md).
- **~49M samples against 32M.**

Budget and split are therefore confounded with configuration in this row.
The sweep-internal trajectory is the cleaner read:

| run | r² | note |
|---|---|---|
| `F1a` | 0.0195 | Stage 0b start |
| `E1a` | 0.0303 | production's hyperparameters, 8M samples |
| `03_s32M` | 0.0316 | selected config, width 0.5 |
| `04_fe4` | 0.0337 | width 1.0, lr 1e-3 |
| `05_lr5e-4` | 0.0347 | width 1.0, lr 5e-4 — best so far |
| **production** | **0.0416** | width 1.0, 49M samples, 80% of trials |

r² has roughly doubled from the Stage 0b start, and the current config at 32M
beats production's config evaluated at the sweep's own budget (`E1a`, 0.0303).
What remains unbeaten is the production model *at its own budget*.

## Discipline: this is reporting, not selection

`fixrsvp` is the true test set. Scoring production on it once, to know where the
sweep stands, is reporting. Letting the answer feed back into width or lr
choices — rescuing a flat ladder rung because "the gap to production is still
open" — is selecting on the test set through the back door. **Ladder decisions
stay on validation BPS.** See the withdrawn budget recommendation in
[03-sample-budget.md](03-sample-budget.md) for the precedent.

## Provenance caveat

`cids_source == "yaml"`, not `"checkpoint"`: this checkpoint predates cid
snapshotting, so its readout population resolves from the session YAMLs rather
than from a snapshot inside the file. `validate_readout_sizes` passed. This is
the same load path `paper/fig3` uses, so the numbers here describe the
production model exactly as the figure does — and it is a live reason the
"never edit `experiments/dataset_configs/sessions/*.yaml`" constraint matters:
editing one would silently repopulate this model's readouts.

## Open question this raises: 64M samples

Experiment 03 stopped at 32M because that was 4× the original budget, not
because gains stopped. Validation BPS went 0.5657 (8M) → 0.5913 (16M) → 0.6072
(32M): increments of +0.0256 and +0.0159, both well above the 0.0080 floor and
diminishing but not exhausted. A 64M rung would plausibly still resolve, and
production's advantage is partly just its 49M samples.

**Deferred until the capacity ladder finishes.** Doubling the budget doubles
every rung's cost (~80 h at width 2.0, ~160 h at width 3.0), so committing to it
before knowing which width is worth paying for would multiply the wrong thing.
Revisit once `06_w2` and `06_w3` have reported.
