# Experiment 04 — the temporal frontend bottleneck

**Question.** The frontend is a learned temporal filter bank: `num_channels`
kernels over a 16-frame (~133 ms at 120 Hz) causal window, gaussian-derivative
init. It has always been 4. Does that bind?

## Why 4, and why the invariance is intentional

4 is a **biological prior**, not a default nobody revisited: midget and
parasol, ON and OFF — the retinal ganglion channels an achromatic stimulus
drives. The temporal bottleneck between retina and cortex is real, and the
frontend was built to approximate it.

`scale_model_config` therefore does *not* scale it, and that is correct
behaviour for the prior: a fixed set of retinal cell types should not grow
with cortical capacity. This was initially misread here as an oversight; it
is not.

## What is nevertheless untested

The prior has never been checked against any alternative. Two things make it
worth checking now:

**Everything temporal passes through it.** For a paper about fixational eye
movements, the temporal dynamics *are* the phenomenon — drift and
microsaccade-driven structure reaches the cortex model only through this
4-dimensional basis.

**Its relative width falls sharply up the ladder.** Even a principled
bottleneck tightens as everything downstream grows:

| width | frontend | stem | blocks | frontend : block |
|---|---|---|---|---|
| 0.25 | 4 | 4 | [16, 32] | 1 : 8 |
| 0.5 | 4 | 4 | [32, 64] | 1 : 16 |
| 1.0 | 4 | 8 | [64, 128] | 1 : 32 |
| 2.0 | 4 | 16 | [128, 256] | 1 : 64 |
| 4.0 | 4 | 32 | [256, 512] | 1 : 128 |

## Design

| run | frontend | width | samples | cost |
|---|---|---|---|---|
| `04_fe4` | 4 | 1.0 | 32M | ~18 h |
| `04_fe8` | 8 | 1.0 | 32M | ~18 h |
| `04_fe16` | 16 | 1.0 | 32M | **only on a large gain at 8** |

Otherwise the configuration experiments 02 and 03 selected: lr 1e-3, batch
128, accumulate 1, homogeneous, no adapter.

**Width 1.0, not 0.5.** The constraint binds harder there (1:32 against 1:16),
so a null at width 0.5 would license nothing higher up the ladder. `04_fe4`
also doubles as the capacity ladder's width-1.0 rung, so the marginal cost of
the comparison is one run rather than two.

**8 before 16.** 8 is still readable as a relaxation of the retinal story —
two temporal subtypes per class — where 16 abandons it. If 4 and 8 are
indistinguishable there is no reason to spend 18 h on 16.

The two configs differ in exactly one line (`num_channels: 4` vs `8`;
verified by diff), and `independent_knobs` reports the single knob
`frontend_channels`.

## Measured before launching

Width 1.0, batch 128, no adapter, homogeneous: **248.3 ms/step, 8.35 GiB,
4.61M params** — 1.67× width 0.5's step time, not the 2× a naive reading of
the parameter count suggests. 127 s/epoch × 488 epochs ≈ 17.2 h plus
validation.

## The fork this creates, recorded before the result

If **8 beats 4**, the finding is not a config win — it is evidence that *the
retinal-bottleneck prior costs accuracy*. That is a claim about the model's
inductive bias and has to be reported as one. It would force a real choice:

- take the accuracy and drop the strict RGC analogy, or
- keep 4 for interpretability and report the measured cost of doing so.

Either is defensible; silently adopting 8 and still describing the frontend as
retina-like is not. Recorded now so the fork is not discovered after the
number arrives and rationalised in whichever direction the number points.

If **4 and 8 are indistinguishable**, the prior is free: it costs nothing and
buys interpretability, which is the strongest possible outcome for it.

## Results

| run | val BPS | Δ vs fe4 | test BPS | CC_norm | r² | hours | verdict |
|---|---|---|---|---|---|---|---|
| `04_fe4` | 0.6059 | — | 0.6137 | 0.613 | 0.0337 | 21.6 | baseline |
| `04_fe8` | 0.6048 | **−0.0011** | 0.6108 | 0.640 | 0.0346 | 22.4 | **unresolved (null)** |

Null on both BPS splits (−0.0011 val, −0.0029 test; 0.14× and 0.41× their
floors). CC_norm and r² are nominally higher but well inside their floors, and
CC_norm is the held-out test set and is not used to select in any case.

**The prior is free.** −0.0011 is 0.14× the floor: doubling the temporal basis
changes nothing. The best outcome available for the retinal story — the
4-channel bottleneck costs no measurable accuracy, so the midget/parasol ×
ON/OFF interpretation survives at no price, and the fork recorded above never
opens. The gate on 16 channels is not met.

**Caveat, weaker than it looks.** This ran at lr 1e-3, which experiment 05
subsequently showed is too high for width 1.0 (see
[05-lr-at-width-1.md](05-lr-at-width-1.md)). But `fe4` and `fe8` differ only
in the frontend filter count and the stem's input channels — negligible
against 4.61M parameters — so both were mistuned identically and the
comparison remains internally controlled. Re-confirm at lr 5e-4 only if the
frontend question becomes load-bearing again.

**What this does not license.** The null is measured at width 1.0. The
frontend's relative width keeps falling up the ladder (1:32 here, 1:128 at
width 4.0), so "the prior is free" is established at this rung only. If the
ladder is later run to width 2.0+, the check is worth repeating there.

Selection on **validation BPS**. Held-out fixrsvp CC_norm is the true test set
and is not used to choose anything — see the correction in
[03-sample-budget.md](03-sample-budget.md), where an earlier budget
recommendation leaned on it and was withdrawn.

## Takeaways

_Pending — `04_fe4` launched 2026-08-08 09:52._
