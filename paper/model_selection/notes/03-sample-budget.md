# Experiment 03 — sample budget

**Question.** Every arm so far has trained on 8M samples against a 7,143,930
sample training set — **1.12 passes**. Nothing is near saturation, and no arm
has overfit: each one's best checkpoint sits at or beside its last epoch. Does
more data still buy anything, and where does it stop?

**Design.** Sample budget only, at the configuration experiment 02 selected:
lr 1e-3, batch 128, accumulate 1, width 0.5, homogeneous, no adapter. Baseline
is `02_lr1e-3_bs128` (0.5657 val BPS).

| run | samples | epochs | passes | launch |
|---|---|---|---|---|
| baseline | 8M | 122 | 1.12 | done |
| `03_s16M` | 16M | 244 | 2.2 | unconditional |
| `03_s32M` | 32M | 488 | 4.5 | **gated** on 2× improving |

4× is the first budget at which overfitting is even plausible, and it is the
most expensive arm in the sweep — hence the gate.

## Read the wall clock with care

**`03_s16M` shared GPU 1 with another user's job for roughly the first half of
its run** — from launch at 04:22 until the other job was moved off at ~10:51,
by which point it was at epoch 121 of 244. Its recorded hours are inflated,
and inflated *unevenly*: the first half ran at roughly half speed and the
second at full speed, so the number is not a rate that can be scaled or
corrected, only discarded.

It is **not** a measurement of what a 2× budget costs. Uncontended that is
~7.1 h, twice the baseline's 3.5 h, because per-sample throughput is flat in
this regime (measured: 1.165 ms/sample at batch 128 against 1.179 at 256).

This matters because `collect.py` prints an `hours` column, and 3.5 h at 8M
next to a contended number at 16M reads as superlinear cost scaling. It is
not. **Contention changes wall clock only** — it does not touch the data
order, the seed, the batch composition or the arithmetic, so val/test BPS and
every other metric are unaffected. Only the timing is contaminated.

Do not use this run to estimate Tier 1's cost. Use 2× the baseline's 3.5 h, or
re-measure on an idle card.

## The interpretive limit, restated

The cosine horizon is `max_epochs`, so a larger budget also stretches the
anneal. That is the right design — a fixed horizon would leave lr at ~0 for
the extra epochs — but these arms confound "more samples" with "slower
schedule" and cannot separate them. Belongs in the model card.

## The free saturation diagnostic — first signal, 2026-08-07

No extra runs needed: watch which epoch holds the best checkpoint. Every arm
through experiment 02 peaked at or beside its final epoch, which is what an
undertrained run looks like.

**`03_s16M` is the first arm whose peak is not at the end.** Best val BPS
0.5913 at epoch 239 of 244; the final validation came in at 0.585. The tail
reads 0.5865 (e195) → 0.5879 (e235) → **0.5913 (e239)** → 0.585 (final).

Read this conservatively. The 0.0063 drop from peak to final sits *inside* the
0.0080 replicate floor, so the honest description is a **plateau over the last
~50 epochs**, not overfitting. What has changed is only that the curve stopped
rising, which is the first thing saturation looks like — and it is exactly why
4× is worth running rather than assumed.

If `03_s32M` also peaks well before its end, that is the budget question
answered: the model saturates somewhere between 16M and 32M samples, and Tier
1 should be sized accordingly rather than at the largest affordable budget.

## Results

| run | val BPS | Δ vs baseline | test BPS | CC_norm | r² | hours | verdict |
|---|---|---|---|---|---|---|---|
| `02_lr1e-3_bs128` | 0.5657 | — | 0.5785 | 0.603 | 0.0312 | 3.5 | baseline |
| `03_s16M` | **0.5913** | **+0.0256** | **0.6000** | 0.606 | 0.0319 | 9.9 ⚠ | resolved better on BPS |
| `03_s32M` | **0.6072** | **+0.0415** | **0.6141** | 0.606 | 0.0316 | 13.6 | resolved better on BPS only |

⚠ contended — see above. Not a cost measurement. `03_s32M`'s 13.6 h ran on a
clear card and *is* usable.

`collect.py` labels `03_s32M` **partial**. It is not: the log reached epoch
487 of 488 and training exited 0. `run_status` reads the last *checkpointed*
epoch, and top-4 retention discarded the final validations (475–487) because
they scored below the peak at 463. The heuristic assumes a run's tail does not
decline — the assumption these arms now violate. Expect more false `partial`
labels as runs start peaking before their end; the flag labels a number, it
never gates one.

## Takeaways

**1. 2× improves BPS and nothing else.**

| metric | 8M | 16M | Δ | floor | resolved? |
|---|---|---|---|---|---|
| val BPS | 0.5657 | 0.5913 | +0.0256 | 0.0080 | yes, 3.2× |
| test BPS | 0.5785 | 0.6000 | +0.0215 | 0.0070 | yes, 3.1× |
| CC_norm | 0.603 | 0.606 | +0.003 | 0.0844 | no, 0.04× |
| r² | 0.0312 | 0.0319 | +0.0007 | 0.0044 | no, 0.16× |

Doubling the data bought in-domain fit and essentially nothing on held-out
fixrsvp CC_norm or single-trial r². Both are flat to within a small fraction
of their floors — this is not a marginal difference, it is no difference.

**This matters for sizing Tier 1, not for the selection metric.** Selection is
settled on BPS. But figures 3 and 4 report CC_norm and single-trial r², and if
4× repeats this pattern then budget beyond ~8-16M samples buys a metric the
figures do not use, while costing GPU-hours linearly. Tier 1 is ~570 GPU-h;
this is the single largest lever on that number found so far.

**2. The plateau and the flat CC_norm are consistent with each other.** The
within-run curve stopped rising over the last ~50 epochs at the same time the
held-out metrics stopped responding to data. Two weak signals pointing the
same way is not a measurement, but it is the shape saturation would take.

**3. 4× answered it: the budget buys BPS and nothing else.**

| metric | 8M | 16M | 32M | Δ 8M→32M | floor | resolved? |
|---|---|---|---|---|---|---|
| val BPS | 0.5657 | 0.5913 | 0.6072 | +0.0415 | 0.0080 | yes, 5.2× |
| test BPS | 0.5785 | 0.6000 | 0.6141 | +0.0356 | 0.0070 | yes, 5.1× |
| CC_norm | 0.603 | 0.606 | 0.606 | +0.003 | 0.0844 | **no, 0.04×** |
| r² | 0.0312 | 0.0319 | 0.0316 | +0.0004 | 0.0044 | **no, 0.09×** |

Quadrupling the data — 3.5 h to 14 h — moved in-domain BPS by 5× its floor and
the two metrics figures 3 and 4 actually report by essentially zero, twice
independently. BPS returns also halve as cost doubles (+0.0256 then +0.0159),
so even on its own terms the curve is bending.

**Recommendation: size Tier 1 at 8M samples**, and spend the saved hours on
capacity or on the replicate groups this sweep has never had. The measured
ladder cost assumed sample-matching to the paper model's ~49M; at 8M that is a
~6× reduction in per-run cost.

**4. The saturation diagnostic fired twice and grew.** `03_s16M` peaked at
epoch 239 of 244 and fell 0.0063 by its last validation; `03_s32M` peaked at
463 of 488 and fell 0.0082. The second is marginally above the 0.0080 floor
where the first was below it. Two arms, same shape, effect growing with
budget — consistent with the flat held-out metrics, and the reason the BPS
gains should not be read as the model still learning something useful.

## The caveat that could overturn this

**Everything here is width 0.5.** Larger models are typically more
data-hungry, and the ladder spans 152×. "8M is enough" is measured at the
bottom rung only, and it is exactly the kind of claim that fails to transfer
upward — the same way experiment 02's lr optimum flipped sign with batch size.

The width sweep must therefore carry a budget check at its *largest* rung, not
inherit 8M on faith. If CC_norm at width 2.0+ responds to budget where width
0.5 did not, this recommendation is wrong and Tier 1 needs the larger budget
after all. Cheapest form: one extra arm at the top width with 4× samples,
compared against the same width at 8M.
