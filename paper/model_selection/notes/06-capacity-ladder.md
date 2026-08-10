# Experiment 06 — the capacity ladder, each rung at its rule-derived lr

**Question.** Does capacity keep buying accuracy once every rung is given the
learning rate an inverse-width rule prescribes?

## Why the lr rule is not optional

Experiment 05 is the reason this experiment has the shape it does. At width
1.0, lr 1e-3 — selected at width 0.5 — produced 0.6059 against width 0.5's
0.6072: 3.7× the parameters for nothing, 0.16× the replicate floor. That reads
as capacity saturating, and it was wrong. At 5e-4 the same model reaches
0.6222, +0.0163 at 2.0× the floor.

So **a ladder run at fixed lr produces a flat curve that is an artifact**, and
it very nearly produced exactly that reading here. Every rung takes

    lr = 5e-4 / width

anchored at width 1.0, the one point where an lr was measured against an
alternative rather than assumed.

The two tuned points the sweep already has both sit on this line — width 0.5 at
1e-3 (0.6072), width 1.0 at 5e-4 (0.6222), a +0.0150 gain at 1.9× the floor.
That is a **consistency check, not a validation**: neither lr was chosen by the
rule, and two points on a line through them is what a one-parameter rule fitted
to an anchor will always give.

## Design

| run | width | params | lr | samples | cost |
|---|---|---|---|---|---|
| `05_lr5e-4` | 1.0 | 4.61M | 5e-4 | 32M | 21.7 h (done, 0.6222) |
| `06_w2` | 2.0 | 17.9M | 2.5e-4 | 32M | ~35.3 h |
| `06_w3` | 3.0 | 39.7M | 1.67e-4 | 32M | ~71.3 h |

Everything else is the configuration experiments 02–05 settled: batch 128,
accumulate 1, frontend 4, homogeneous, adapter off, seed 201, 488 epochs.
Serial on GPU 1, ~107 h total.

The frontend stays at 4 up the ladder — `scale_model_config` does not scale it,
which is correct for the retinal prior (midget/parasol × ON/OFF). The
`_fe4` configs generated for these rungs are byte-identical to the plain
`widthN_noadapter.yaml` apart from a header comment (verified by diff); they
exist so the spec states the pinned frontend explicitly rather than by default,
which keeps it out of the labels.

## Measured before launching

From the capacity probe (4 datasets loaded), scaled by the 1.5× measured at
width 1.0 between the probe (8.35 GiB) and a real 30-dataset run (12.8 GiB):

| width | probe GiB | est. real GiB | ms/step |
|---|---|---|---|
| 2.0 | 16.4 | ~25 | 509 |
| 3.0 | 24.7 | ~37 | 1027 |
| 4.0 | 33.1 | **~50** | 1363 |

Both rungs here fit inside 49 GiB. **Width 4.0 does not** at batch 128, and is
the rung that will force a batch-size change or gradient checkpointing. Noted
now so that a future OOM is a prediction confirmed rather than a surprise.

## The ambiguity this design cannot resolve, recorded before the result

Each rung moves **two** knobs against the last — width and lr. That makes the
outcomes asymmetric:

- **The curve keeps climbing** → capacity pays, under a rule that is at worst
  approximately right. Unambiguous: no lr error explains a model doing *better*
  than the rung below at a rule-set lr.
- **The curve goes flat** → genuinely ambiguous. Either capacity has saturated,
  or the rule went wrong at that rung. This is precisely the confusion
  experiment 05 resolved at width 1.0, and this design cannot resolve it again.

Only a bracket separates them: **width 2.0 at 1.25e-4**, half the rule's
prescription, ~35 h. It is deliberately not queued. It is worth its cost
exactly when the curve goes flat and premature if the curve climbs, and running
the rungs serially means `06_w2`'s number arrives ~35 h before `06_w3` starts —
a natural decision point either way.

**A flat curve must not be reported as saturation without the bracket.** Writing
this down before the number arrives is the point: experiment 05's history makes
"capacity saturated" the tempting reading and it is the one that was already
wrong once.

## What to watch besides the endpoint

**The crossover diagnostic.** Experiment 05 was solved by validation *curves*,
not endpoints: too-large a step helps early, hurts mid-training, and the damage
vanishes as lr anneals. If `06_w2` leads early and trails mid-run against
`05_lr5e-4`, that is the same signature and says 2.5e-4 is still too high —
readable long before the run ends.

**The saturation pattern.** Every long run in experiments 03–05 peaked before
its final epoch (`05_lr5e-4`: peak 0.6222 at epoch 471 of 488, final 0.615). A
fifth occurrence makes it a property of the schedule rather than a coincidence,
and would justify revisiting the cosine horizon.

Selection on **validation BPS**. Held-out fixrsvp CC_norm is the true test set
and selects nothing — see the withdrawn recommendation in
[03-sample-budget.md](03-sample-budget.md).

## Results

_Pending._

## Takeaways

_Pending._
