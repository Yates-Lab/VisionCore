# Stage 0b — running notes

Homogeneous batching, no adapter, width 0.5. Arms and rationale are in
`launch.py`; this file records what the runs actually showed, in the order it
was learned. Nothing here is a paper result.

Sweep started 2026-08-05 15:49, serial on GPU 1, order
`F1a F1b F2c F1c F2a F3a F3c`, train-then-evaluate each.

---

## Dataset loading costs ~3 min, not ~19

Measured on F1a: **3 min 05 s** from process start to the first training step,
covering Python and torch import, wandb init, and preprocessing all 30
sessions. The figure carried in `NEXT_SESSION.md` and repeated through the
sweep planning was ~19 min, about 6× too high. (Caveat: this box has read the
same data repeatedly today, so the OS page cache is warm; a cold first read may
be slower. It is not 19 min under working conditions.)

**Nothing is shared between runs.** Each arm pays three independent
preprocessing passes over the same sessions:

| pass | entry point | stimulus types |
|---|---|---|
| train | `launch.py` → `MultiDatasetDM.setup("fit")` | backimage, gaborium, gratings |
| eval 1 | `evaluate.py` → `build_test_datamodule` → `setup("test")` | the same three |
| eval 2 | `score_fixrsvp` → `load_single_dataset` → `prepare_data`, per session | those three **+ fixrsvp** |

There is no disk cache for preprocessed tensors; the only caching in
`models/data/loading.py` is for STAs/STEs, and the 240→120 Hz downsample is
redone every pass.

Two available savings, neither taken during the sweep because both sit in the
data path that `data_census.py` gates and arms were already in flight:

1. **Eval pass 2 loads a strict superset of pass 1.** One load with all four
   stimulus types could serve both, entirely inside `evaluate.py`. Removes a
   third of the loading with no cross-process machinery.
2. **Sharing across arms.** A driver holding one datamodule and building a
   fresh model per arm would pay the cost once instead of seven times. Needs
   per-arm re-seeding and care about CUDA fragmentation across seven builds.

At ~3 min a pass this is worth about **1 h across Stage 0b's ~27 h**, so it is
a cleanup, not a bottleneck. Priority accordingly.

---

## Width 0.5 *appears* to plateau at 8M samples — but see F2a: it was update-limited, not data-limited

F1a's validation curve, every 4th epoch:

```
0.065  0.288  0.358  0.414  0.434  0.441  0.470  0.480
0.503  0.501  0.508  0.510  0.526  0.523  0.524
```

The last three validations sit within ±0.003 of each other after mid-run steps
of 0.01–0.02, so the run has flattened. Both width-1.0 arms behaved the other
way: E1a was still climbing ~0.007 per validation at its horizon, and E1b's
last three were 0.575 / 0.571 / 0.575 with no plateau before them.

**Superseded by F2a — read the correction below before using this.**

The original reading was "samples-to-convergence scales with capacity, so 8M is
adequate at width 0.5". F2a refutes it: same architecture, same 8M samples, but
4x more optimizer steps (effective batch 256 instead of 1024), and it reaches
**0.5473** against the baseline's 0.5261. The plateau was not a data or capacity
ceiling; it was an **optimization** ceiling set by the number of updates.

So the flat tail of an F1 curve does not mean the budget is sufficient. It means
that configuration has stopped improving at that update count. F2a plateaus too,
but ~0.021 higher.

Confirmed across all three replicates: F1a, F1b and F1c each peaked at **epoch
51** and each ended flat. The plateau is a property of the configuration, not of
one seed.

**The BPS floors grew with the third replicate**: val BPS range 0.0044 at
n = 2 → **0.0080** at n = 3 (+82%), test BPS 0.0035 → 0.0070 (+100%). A two-run
"spread" is a single difference and understates systematically, which is what
`stability.py` warns about at n = 2 — now with a number attached. CC_norm was
the exception: F1c landed between the other two, leaving the range at 0.0844
and lowering the sd. Even at n = 3 these are lower bounds.

Note F1a's slow start — 0.065 at epoch 3, against E1a's 0.295 and E1b's 0.224.
Not interpreted: width, the missing adapter, and seed all differ, and by epoch
15 it is on a normal trajectory.

---

## F1a's generalisation drop is confounded with width — F0 added to resolve it

F1a evaluated well below E1b, but *two* things differ between them (width
1.0 → 0.5, adapter on → off), so neither is isolated. This is the cost of
adopting adapter-off without a dedicated arm.

| | E1b (w1.0, adapter on) | F1a (w0.5, adapter off) |
|---|---|---|
| test BPS | 0.5858 | 0.5351 (−8.7%) |
| fixrsvp CC_norm | 0.639 | **0.465 (−27%)** |
| single-trial r² | 0.0302 | 0.0195 (−35%) |

The asymmetry is what makes it worth a run rather than a shrug: **in-domain BPS
fell 8.7% while out-of-domain CC_norm fell 27%.** Pure capacity loss should move
both together. A loss concentrated in generalisation is what removing a fixed
σ = 1 Gaussian pre-blur would look like — less smoothing, more high-frequency
fitting of the training conditions, worse transfer to `fixrsvp`. Hypothesis, not
a finding.

Scale is the reason it can't wait for Tier 1: −0.174 CC_norm dwarfs the +0.050
that motivated switching to homogeneous batching at all.

**F0** (adapter ON, width 0.5, homogeneous, seed 201) differs from F1a in
exactly one thing and settles it. Run 2026-08-05 on GPU 0, parallel to the
sweep on GPU 1.

### Result: the adapter does nothing detectable. Removing it was right.

| | F0 (adapter ON) | F1a | F1b | adapter-off mean |
|---|---|---|---|---|
| val BPS | 0.5235 | 0.5264 | 0.5220 | 0.5242 |
| test BPS | 0.5356 | 0.5351 | 0.5316 | 0.5333 |
| fixrsvp CC_norm | **0.530** | 0.465 | 0.549 | 0.507 |

F0 lands *between* the two adapter-off replicates on CC_norm. Every delta is
`unresolved`: Δval +0.0001 against a 0.0044 floor, Δtest +0.0023 against
0.0035, ΔCC_norm +0.023 against 0.084.

So the −0.174 CC_norm gap between F1a and E1b is **not** the adapter, and by
elimination is mostly the width drop from 1.0 to 0.5. The σ = 1 pre-blur
hypothesis is not supported — dropping the blur cost nothing measurable.

Two honest limits on that conclusion:

- **n = 1 against a 2-replicate baseline.** The test can only exclude an
  adapter effect larger than roughly the 0.084 CC_norm floor. A smaller real
  effect would be invisible. On BPS, where the floor is 0.0035, the exclusion
  is much tighter and the null is solid.
- It says nothing about the adapter at width 1.0, where the readouts have more
  capacity to absorb per-session scaling.

**Consequence:** keep the adapter off, and treat CC_norm differences between
widths as capacity effects rather than architecture effects.

The F2/F3 deltas are unaffected either way — every F1/F2/F3 arm shares one
architecture, so the tuning screen stays internally valid regardless of how the
adapter question resolves.

---

## CC_norm is ~12x noisier than BPS across seeds — and that undermines the E1 decision

F1a and F1b are the *same configuration* at seeds 201 and 202. Nothing differs
but weight initialisation and GPU nondeterminism (not even data order, since
`split_inds_by_trial*` re-seeds globally).

Final three-replicate floor (F1a/F1b/F1c, seeds 201/202/203):

| metric | F1a | F1b | F1c | mean | range | sd |
|---|---|---|---|---|---|---|
| val BPS | 0.5264 | 0.5220 | 0.5300 | 0.5261 | 0.0080 | 0.0040 |
| test BPS | 0.5351 | 0.5316 | 0.5386 | 0.5351 | 0.0070 | 0.0035 |
| **fixrsvp CC_norm** | **0.465** | **0.549** | **0.505** | 0.5065 | **0.0844** | 0.0422 |
| single-trial r² | 0.0195 | 0.0239 | 0.0236 | 0.0223 | 0.0044 | 0.0025 |

**Held-out CC_norm moves 0.084 between identical runs.** Relative to its own
mean that is ~17%, against ~1.3% for test BPS — roughly 12x noisier.

The CC_norm range did *not* widen when the third replicate arrived (F1c landed
between the other two, so the extremes were unchanged and the sd fell from
0.0597 to 0.0422). val and test BPS ranges did widen, by ~82% and ~100%. With
n = 3 all three should still be read as lower bounds.

### What this invalidates

Stage 0b adopted homogeneous batching because **E1b beat E1a by +0.050 CC_norm**
while losing 0.023 test BPS. That +0.050 is **inside** the 0.084 seed-to-seed
spread of the metric it was measured on. The evidence for that decision does not
survive its own noise floor.

The per-session consistency that made it look solid (E1b higher in 16 of 17
sessions) does not rescue it. That was flagged at the time — a replicate at a
different seed would also shift most sessions together — and this is the
quantified version of exactly that caveat.

Note the throughput half of the argument had already collapsed: the measured
step-time difference at width 1.0 is ~2%, and the 12% wall-clock gap between the
two runs was box contention.

### What it does *not* invalidate

- **Figures 3 and 4.** Their CC_norm comparisons are *within* one model
  (intact vs extraretinal-zeroed vs stabilized, same weights), so seed spread
  does not enter. A single model's reported CC_norm is a description of that
  model, not a comparison between models.
- **Selection on BPS.** `protocol.SELECTION_METRIC` is validation BPS, and this
  is a strong post-hoc justification for that choice: BPS is stable across
  seeds at this budget, CC_norm is not.
- **The F2/F3 tuning screen.** Those are single-knob comparisons within one
  family, and they will be read against this floor.

### What to do about it

1. **Read every CC_norm delta against 0.084, not against zero.** Most will be
   unresolvable. Tuning decisions have to rest on BPS.
2. **Any model-vs-model CC_norm claim needs replicates.** One run per arm
   cannot support one, which is what the E1 comparison tried to do.
3. **Revisiting the batching choice needs replicated E1a/E1b at width 1.0** —
   about 4 runs, ~22 GPU-h. Until then homogeneous batching is adopted on
   evidence that has not survived, though nothing yet shows it is *worse*.

---

## F2c is clearly worse — but F2 confounds gradient averaging with update count

F2c (effective batch 4096) reached val BPS **0.4822** against the baseline mean
0.5242: **Δ −0.042**, about 9.5x the 0.0044 floor. Resolved, and in the opposite
direction to the gradient-variance hypothesis that motivated the arm.

It cannot be read that way, because of how the arm is built:

| arm | eff. batch | accumulate | steps/epoch | total steps | lr |
|---|---|---|---|---|---|
| F2a | 256 | 1 | 512 | 31,232 | 1e-3 |
| F1a/b/c | 1024 | 4 | 128 | 7,808 | 1e-3 |
| F2c | 4096 | 16 | 32 | **1,952** | 1e-3 |

The sample budget is fixed at 8M, so raising the effective batch cuts optimizer
steps proportionally. **F2c took 4x fewer updates than the baseline at the same
learning rate**, and the F2 arms span 16x in update count. Undertraining is a
sufficient explanation for its deficit, entirely separate from how many sessions
each gradient averages.

This is a design flaw inherited from Stage 0's E2, where it had the same
problem. Fixed-sample-budget batch sweeps need the learning rate scaled with
the batch (linear or sqrt) or they measure step count.

**Prediction to check against F2a:** if update count dominates, F2a (4x *more*
steps) should beat the baseline, giving a monotonic F2a > F1 > F2c ordering that
tracks step count rather than batch composition. If instead F2a is flat or worse,
something other than step count is at work and the arm becomes interpretable.

**F3c is a partial control.** It runs lr 3e-3 at the baseline batch — close to
the 4e-3 that linear scaling would prescribe for F2c's 4x batch. If 3e-3 is
already worse at eff batch 1024, F2c at a scaled lr probably would not recover
either.

**To settle it properly** needs one crossed arm: effective batch 4096 *with*
lr 4e-3 (~3.3 h). That separates the two cleanly and is worth adding once the
declared arms finish.

---

## F2a settles it: the model is update-limited. Set accumulation to 1.

F2a (effective batch 256, accumulate 1) reached val BPS **0.5473** — Δ **+0.0212**
against the baseline mean, 2.65x the 0.0080 floor. Resolved, and better.

With all three F2 arms in, the ordering is monotonic in optimizer steps and
nothing else:

| arm | eff. batch | accumulate | optimizer steps | val BPS | Δ vs baseline |
|---|---|---|---|---|---|
| F2a | 256 | 1 | 31,232 | **0.5473** | +0.0212 (resolved) |
| F1a/b/c | 1024 | 4 | 7,808 | 0.5261 | baseline |
| F2c | 4096 | 16 | 1,952 | 0.4822 | −0.0439 (resolved) |

**What this establishes.** At a fixed 8M-sample budget and fixed lr, more
optimizer steps is strictly better across a 16x range. F2c's deficit was
undertraining, so it is *not* evidence about gradient composition, and the
gradient-variance hypothesis remains untested rather than refuted.

**It costs nothing.** Accumulation does not change sample throughput: F2a took
3 h 15 min, the same as every other arm. Same wall clock, same samples, better
model. Set `accumulate = 1` unless something else argues against it.

Held-out evaluation agrees, and shows the floor working as intended:

| metric | F2a | baseline mean | Δ | floor | verdict |
|---|---|---|---|---|---|
| val BPS | 0.5473 | 0.5261 | +0.0212 | 0.0080 | resolved |
| test BPS | 0.5560 | 0.5351 | +0.0209 | 0.0070 | resolved |
| CC_norm | 0.583 | 0.5065 | +0.077 | 0.0844 | **unresolved** |
| single-trial r² | 0.0277 | 0.0223 | +0.0054 | 0.0044 | marginal |

The same change is resolved on both BPS measures and unresolved on CC_norm,
purely because CC_norm's floor is ~12x wider. A CC_norm-only comparison — which
is what the E1 decision was — could not have seen a real +0.077 improvement.

**It supersedes the plateau reading above.** F1's flat tail was an optimization
ceiling at 7,808 updates, not a data ceiling — F2a keeps climbing past it on the
same budget.

**Open, and worth one arm.** accumulate = 1 at batch_size 256 is the floor of
what the current arms explore, but the trend does not obviously stop there:
halving `batch_size` to 128 at accumulate 1 doubles updates again (~62k) for the
same samples. Whether the curve keeps rising or turns over is unmeasured, and it
is the cheapest remaining question in the family.

**Still needed to test the original hypothesis:** effective batch 4096 *with*
lr 4e-3 (linear scaling), which holds update count's confound fixed and asks
whether averaging more sessions per update helps once the learning rate
compensates.

---

## Summary — Stage 0b complete (2026-08-06 18:31)

Eight runs, ~27 h, all trained and evaluated cleanly. Verdicts are against the
three-replicate floor (val 0.0080, test 0.0070, CC_norm 0.0844).

| arm | knob | Δ val BPS | verdict |
|---|---|---|---|
| **F3c** | **lr 3e-3** | **+0.0328** | **resolved better — best arm** |
| F2a | accumulate 1 (eff batch 256) | +0.0212 | resolved better |
| F0 | adapter ON | −0.0018 | unresolved (null) |
| F2c | eff batch 4096 | −0.0439 | resolved worse |
| F3a | lr 3e-4 | −0.0964 | resolved worse |

### What is settled

1. **The adapter does nothing.** Keep it off. (F0, null on every metric.)
2. **This configuration is optimization-limited, not data-limited.** Both levers
   on total optimization progress move it a long way in both directions:
   update count monotonically across 16x (F2a/F1/F2c) and learning rate
   monotonically across 10x (F3a/F1/F3c). Neither has turned over at the tested
   extremes.
3. **`accumulate = 1` is free.** Accumulation does not change sample
   throughput, so F2a's +0.021 cost nothing in wall clock.
4. **Learning rate is the stronger single lever.** 3x the lr (+0.0328) beats 4x
   the updates (+0.0212).
5. **CC_norm cannot support single-run model comparisons.** Floor 0.0844, ~12x
   test BPS's. Note F2a and F3c both improve CC_norm by ~+0.077 and *neither*
   is resolved, while both are resolved on BPS at 3-5x their floor. Select on
   BPS.

### What is not settled

- **The gradient-variance hypothesis is untested.** F2 confounded batch
  composition with update count, so nothing here speaks to whether averaging
  more sessions per update helps at matched optimization.
- **Neither lever's ceiling is known.** lr 3e-3 is the largest tested and still
  improving; accumulate 1 at bs 256 is the most updates tested.
- **They have never been combined.** No arm runs lr 3e-3 *with* accumulate 1,
  and they are the two winners.
- **Homogeneous vs cross-session.** The evidence that chose it (E1's +0.050
  CC_norm) is inside CC_norm's floor. Nothing shows it is worse; it is simply
  unevidenced.
- **Transfer to width 1.0+.** Everything here is width 0.5. A setting tuned at
  one rung of a 152x ladder need not hold at another, which is what C1 exists
  for.

### Recommended next arms, in value order

| arm | change | cost | why |
|---|---|---|---|
| F4 | lr 3e-3 + accumulate 1 | 3.3 h | The two winners composed. Untested and the best-guess configuration. |
| F5 | lr 1e-2, accumulate 1 | 3.3 h | Finds where lr turns over. Cheap, and 3e-3 was still climbing. |
| F6 | eff batch 4096 + lr 4e-3 | 3.3 h | Holds update count's confound fixed; the actual test of the gradient-variance hypothesis. |
| C-arm | best config at width 1.0 | 5.5 h | Transfer check before Tier 1 commits. |

---

## Arm results

Filled in as arms land. Read every delta against the replicate floor from
F1a/F1b/F1c — `stability.py` refuses to print a verdict until at least two
exist, and the floor understates true spread because replicates share their
data order.

| arm | val BPS | test BPS | fixrsvp CC_norm | single-trial r² | hours | note |
|---|---|---|---|---|---|---|
| F0  | **0.5235** | **0.5356** | **0.530** | **0.0218** | 3.34 | adapter ON control — no detectable effect |
| F1a | **0.5264** | **0.5351** | **0.465** | **0.0195** | 3.25 | baseline R1 (best ckpt epoch 51) |
| F1b | **0.5220** | **0.5316** | **0.549** | **0.0239** | 3.26 | baseline R2 |
| F2c | **0.4822** | **0.4918** | **0.415** | **0.0128** | 3.25 | eff batch 4096 — resolved worse, but confounded with 4x fewer updates |
| F1c | **0.5300** | **0.5386** | **0.505** | **0.0236** | 3.24 | baseline R3 |
| F2a | **0.5473** | **0.5560** | **0.583** | **0.0277** | 3.25 | eff batch 256 / accumulate 1 — **best arm**, resolved better on both BPS |
| F3a | **0.4297** | **0.4401** | **0.428** | **0.0136** | 3.25 | lr 3e-4 — resolved much worse (12x floor) |
| F3c | **0.5589** | **0.5689** | **0.583** | **0.0251** | 3.24 | lr 3e-3 — **best arm**, resolved better (4.1x floor) |

### Carried over from Stage 0, for reference only

Not a baseline for anything here — different architecture (adapter on) and
batching (cross-session), and width 1.0 rather than 0.5.

| arm | val BPS | test BPS | fixrsvp CC_norm | single-trial r² |
|---|---|---|---|---|
| E1a | 0.5958 | 0.6090 | 0.589 | 0.0303 |
| E1b | 0.5752 | 0.5858 | 0.639 | 0.0302 |
