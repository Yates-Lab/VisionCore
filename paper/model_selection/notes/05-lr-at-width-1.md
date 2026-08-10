# Experiment 05 — is the width-1.0 null an lr artifact?

**The result that prompted this.** At a matched 32M samples, lr 1e-3, batch
128, accumulate 1, frontend 4:

| run | width | params | val BPS |
|---|---|---|---|
| `03_s32M` | 0.5 | 1.23M | 0.6072 |
| `04_fe4` | 1.0 | 4.61M | 0.6059 |
| Δ | 2× | **3.7×** | **−0.0013** (0.16× floor) |

3.7× the parameters bought nothing. Read naively that says capacity has
saturated — and it would condemn Tier 1, which is a capacity ladder.

**Why that reading is premature.** lr 1e-3 was selected at width 0.5, and
experiment 02 established that this optimum does not transport across regimes:
it flipped *sign* between effective batch 1024 and 256. Larger models
conventionally want smaller steps, and nothing here has tested that.

## The evidence: validation curves, not endpoints

| epoch | w0.5 | w1.0 | diff |
|---|---|---|---|
| 3 | 0.038 | 0.047 | +0.009 |
| 7 | 0.228 | 0.298 | **+0.070** |
| 15 | 0.365 | 0.403 | +0.038 |
| 31 | 0.438 | 0.472 | +0.034 |
| 63 | 0.499 | 0.491 | −0.008 |
| 95 | 0.521 | 0.500 | **−0.021** |
| 127 | 0.543 | 0.530 | −0.013 |
| 191 | 0.559 | 0.545 | −0.014 |
| 255 | 0.563 | 0.547 | −0.016 |
| 319 | 0.581 | 0.577 | −0.004 |
| 383 | 0.598 | 0.595 | −0.003 |
| 479 | 0.604 | 0.599 | −0.005 |

Width 1.0 leads substantially early, crosses over near epoch 63, trails by
~0.02 through mid-training, then converges to parity as the run ends.

**This is what too-large a step size looks like under cosine annealing.** Big
steps help while the loss surface is coarse, hurt once fine structure matters,
and the damage disappears as lr anneals toward zero — recovery to parity
exactly as lr → 0 is the signature.

It also discriminates against the alternatives. A frontend throttle or
saturated capacity predicts **parity throughout**; neither explains a larger
model being measurably *worse* in mid-training. And the early lead shows the
extra capacity is usable when the optimizer can exploit it.

**A diagnostic that did not work, recorded so it is not retried.** Comparing
`train_loss_step` between the two runs looked promising — same seed, same data
order, so the comparison is paired — and the two matched to ~0.001 at every
sampled epoch. But the trajectory is non-monotonic (0.222 → 0.285 → 0.207 →
0.334 → 0.153): under homogeneous batching each step is a single session, so
the logged step loss is dominated by which session was drawn, not by learning
progress. The only thing it supports is a weak argument against overfitting —
a 4×-capacity model that was overfitting should show *lower* training loss,
and it shows none.

## Design

| run | lr | width | samples | cost |
|---|---|---|---|---|
| `04_fe4` | 1e-3 | 1.0 | 32M | 22.3 h (done) |
| `05_lr5e-4` | 5e-4 | 1.0 | 32M | ~22 h |
| `05_lr3e-4` | 3e-4 | 1.0 | 32M | **only if 5e-4 helps** |

Identical in every other respect — same config file
(`width1_noadapter_fe4.yaml`), so `independent_knobs` reports the single knob
`lr`.

**Why 5e-4:** the inverse-width halving for a doubled width. Experiment 01's
lr 3e-4 arm collapsed to 0.4297, but it ran at effective batch 1024 with
31,232 optimizer steps; this configuration takes **249,856**. With 8× the
updates a smaller step has room to work, so that collapse does not transfer.

Run on GPU 0 in parallel with `04_fe8` on GPU 1, so the frontend and lr
questions resolve together rather than a day apart.

## What each outcome means

- **5e-4 beats 1e-3 and clears width 0.5's 0.6072** → the width null was an lr
  artifact, capacity does help, and Tier 1's ladder is worth running. Every
  width rung then needs its own lr, which changes the ladder's design and
  cost.
- **5e-4 matches 1e-3** → lr is not the explanation; attention returns to the
  frontend (`04_fe8`) and then to genuine capacity saturation.
- **5e-4 is worse** → 1e-3 was already at or below the optimum for width 1.0,
  and the width null needs a different explanation entirely.

## Interaction with experiment 04

`04_fe8` remains a valid comparison against `04_fe4` — same lr, nearly the
same model size, so the frontend contrast is internally consistent. What
changes is its *interpretation*: if lr is mistuned at width 1.0, the frontend
question will have been answered at a suboptimal operating point, and a null
there is weaker evidence than it appears.

## Results

| run | val BPS | Δ vs 04_fe4 | test BPS | CC_norm | r² | hours | verdict |
|---|---|---|---|---|---|---|---|
| `04_fe4` | 0.6059 | — | 0.6137 | 0.613 | 0.0337 | 21.6 | baseline |
| `05_lr5e-4` | **0.6222** | **+0.0163** | **0.6280** | 0.658 | 0.0347 | 21.7 | **resolved better** |

Resolved on both BPS splits (+0.0163 val at 2.0× floor, +0.0143 test at 2.0×).
CC_norm 0.658 is the highest in the sweep but is *not* used to select — it is
the held-out test set; recorded only. r² +0.0010 is inside its floor.

## Takeaways

**1. Confirmed: lr 1e-3 was too high at width 1.0.** +0.0163 at 2.0× the
floor, on a single-knob comparison. The crossover diagnosis was right, and
0.6222 is the best validation BPS of the sweep.

**2. The capacity claim is weaker than the lr claim, and must not be
conflated with it.** It is tempting to read 0.6222 against width 0.5's 0.6072
as "capacity helps after all", but that comparison moves *two* knobs — width
and lr. Width 0.5 was never lr-tuned at 32M either; it was only ever run at
1e-3. If lower lr also lifts width 0.5, both rise and the capacity question
stays open.

The conventional picture — larger models want smaller steps — implies 1e-3
sits nearer width 0.5's optimum than width 1.0's, which would make the
comparison roughly fair. That is an assumption, not a measurement. **What is
established is the lr effect. The capacity claim rests on it and is
suggestive only.**

**3. The design consequence for Tier 1.** A capacity ladder run at fixed lr
would produce a flat curve that is an artifact — which is precisely what
happened at width 1.0 and was nearly read as capacity saturation. The ladder
therefore needs an lr rule per rung (inverse-width scaling, or a probe at each
rung), which changes both its design and its cost. This is the single most
consequential finding for Tier 1 in the sweep.

**4. Two points give a direction, not an optimum.** 1e-3 → 5e-4 improves;
whether 5e-4 *is* the optimum or merely better is untested, and an lr rule
needs the optimum, not the direction. `05_lr3e-4` is the natural next probe.

**5. The saturation pattern recurred a fourth time.** Peak 0.6222 at epoch 471
of 488, final validation 0.615 — a 0.0072 drop, just inside the floor. Every
long run in experiments 03-05 has now peaked before its final epoch.
