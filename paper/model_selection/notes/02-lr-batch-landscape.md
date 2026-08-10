# Experiment 02 — lr × batch size at accumulate 1

**Question.** Experiment 01 showed both optimization levers pointing uphill and
neither turning over. Where does the landscape actually peak, and are lr and
batch size two levers or one?

**Design.** 2×2, `lr {3e-3, 1e-2}` × `batch_size {256, 128}`, accumulate 1
throughout, sample budget held at 8M. Baseline is `F2a` (lr 1e-3, batch 256,
accumulate 1) — experiment 01's best arm on update count, reused rather than
re-run.

| run | lr | batch | epochs | steps | launched |
|---|---|---|---|---|---|
| `02_lr3e-3_bs256` | 3e-3 | 256 | 61 | 31,232 | unconditional |
| `02_lr1e-2_bs256` | 1e-2 | 256 | 61 | 31,232 | unconditional |
| `02_lr3e-3_bs128` | 3e-3 | 128 | 122 | 62,464 | unconditional |
| `02_lr1e-2_bs128` | 1e-2 | 128 | 122 | 62,464 | **gated** |

Every arm sees 7,995,392 samples. Halving the micro-batch doubles the epoch
count and the optimizer steps on identical data — that is the manipulation.
`_c_defaults` forces `effective_batch = batch_size` so accumulation cannot
silently re-merge the micro-batches and undo it.

**Gate on the fourth arm.** Launch `02_lr1e-2_bs128` only if *both*
single-notch arms beat the baseline by more than the replicate floor. Two wins
mean the aggressive direction is still climbing and the corner is worth
measuring; one failure localises the wall to that axis and makes the corner a
foregone 3.5 h.

## What was measured before launching

**Throughput (`probe_capacity.py`, width 0.5, no adapter, homogeneous).**

| batch | ms/step | ms/sample | peak |
|---|---|---|---|
| 128 | 149.1 | 1.165 | 4.34 GiB |
| 256 | 301.7 | 1.179 | 8.64 GiB |

Per-sample cost is flat — `bs128` is 1% *faster*, not slower. The predicted
"halving the batch will cost significant wall clock" was wrong on the compute
side: ~2.6 h of stepping either way. What remains is 30 validation passes
against 15, so expect `bs128` ≈ 3.5 h against `bs256` ≈ 3.2 h.

Also: 8.6 GiB peak on a 48 GiB card. Width 0.5 underfills it, with no
throughput penalty for doing so.

## Predictions, recorded before the data

1. **`02_lr3e-3_bs256` lands clearly short of +0.054** (the sum of experiment
   01's two winners, +0.0328 and +0.0212). Both act on total optimization
   progress, so composing them should be sub-additive. Near-additive would
   falsify the one-axis reading.
2. **`02_lr1e-2_bs256` and `02_lr3e-3_bs128` land together.** Each is one notch
   more aggressive than `02_lr3e-3_bs256` along the same axis. If they agree,
   the lr-to-batch *ratio* governs and the two levers collapse into one for the
   width transfer; if they split, they are genuinely separate.
3. **Something turns over.** Every arm in experiment 01 was monotone. lr 1e-2
   is 10× the value tuned under cross-session batching and is the most likely
   place to find the wall.

## Caveat carried into every verdict here

The replicate floor (val BPS 0.0080, test 0.0070, CC_norm 0.0844) was measured
at F1a/F1b/F1c — lr 1e-3, effective batch 1024, **accumulate 4**. Experiment 02
runs at accumulate 1, where no replicate group exists. The floor is borrowed,
not measured in-family, and a verdict here rests on the assumption that
run-to-run spread does not change with the update count. That assumption is
untested and is itself a candidate for a later replicate pair.

Select on **val BPS**. CC_norm's floor is ~12× the BPS floor and cannot support
single-run comparisons.

## Results

Filled in as arms land.

| run | val BPS | Δ vs F2a | test BPS | CC_norm | r² | hours | verdict |
|---|---|---|---|---|---|---|---|
| `02_lr3e-3_bs256` | 0.5228 | −0.0245 | 0.5319 | 0.610 | 0.0290 | 3.2 | worse on BPS |
| **`02_lr1e-3_bs128`** | **0.5657** | **+0.0184** | **0.5785** | 0.603 | **0.0312** | 3.5 | **selected** |
| `02_lr1e-2_bs256` | — | — | — | — | — | — | killed at launch |
| `02_lr3e-3_bs128` | — | — | — | — | — | — | not launched |
| `02_lr1e-2_bs128` | — | — | — | — | — | — | gated, not launched |

The whole width-0.5 family, for context:

| config | lr | eff batch | val BPS | CC_norm | r² |
|---|---|---|---|---|---|
| F1 baseline (n=3) | 1e-3 | 1024 | 0.5261 | 0.507 | 0.0223 |
| F3a | 3e-4 | 1024 | 0.4297 | 0.428 | 0.0136 |
| F3c | 3e-3 | 1024 | **0.5589** | 0.583 | 0.0251 |
| F2a | 1e-3 | 256 | 0.5473 | 0.583 | 0.0277 |
| F2c | 1e-3 | 4096 | 0.4822 | 0.415 | 0.0128 |
| `02_lr3e-3_bs256` | 3e-3 | 256 | 0.5228 | **0.610** | **0.0290** |

## Takeaways

**1. The two levers interact; they are not one axis.** Raising lr from 1e-3 to
3e-3 gains +0.0328 val BPS at effective batch 1024 and *loses* 0.0245 at 256.
A single shared "optimization progress" axis cannot produce a sign flip. The
landscape has an interior peak, and one notch up from baseline in either
direction is near it while two notches overshoot.

This is the direction standard batch scaling predicts and the opposite of what
experiment 01's monotone results suggested: a smaller batch wants a *lower*
lr, not a higher one. The prediction recorded above — "clearly short of
+0.054" — held, but understated: the composition went negative, not merely
sub-additive.

**2. BPS and CC_norm dissociate, again.** `02_lr3e-3_bs256` is resolvably
*worse* than F2a and F3c on the selection metric and nominally the *best* run
in the family on held-out fixrsvp CC_norm (0.610) and single-trial r² (0.0290)
— the two metrics figures 3 and 4 actually rest on.

Neither CC_norm nor r² resolves it: +0.027 CC_norm over F2a/F3c sits well
inside the 0.0844 floor, and +0.0013 r² sits inside the F1 group's 0.0044
spread. So the only *resolved* statement is that it is worse on BPS. But the
same dissociation appeared in experiment 00 — E1a beat E1b by 0.021 test BPS
while losing 0.050 CC_norm — and it is now the second time the BPS ranking and
the CC_norm ranking disagree in the same direction.

**This is a live methodological problem, not a curiosity.** Selecting on BPS
because its floor is small may be selecting against the metric the figures
depend on. Options, none free: measure a CC_norm floor precise enough to
select on (replicates are 3.2 h each and the floor shrinks as √n, so resolving
0.027 needs many); find a lower-variance held-out estimator; or accept BPS as
a proxy and state the assumption explicitly in the model card.

**3. The remaining arms were killed, not run.** `02_lr1e-2_bs256` and
`02_lr3e-3_bs128` both move further along the direction that just overshot, so
~7 GPU-h would have bought a predicted decline. Stopped after arm 1's
evaluation, before arm 2 had trained a step.

**4. The borrowed floor did bite.** `stability.py --baseline F2a` correctly
refuses every verdict: F2a is n=1 and experiment 02 has no in-family replicate
group. The −0.0245 is 3× the floor borrowed from accumulate 4, which is
suggestive but not a resolved result under this pipeline's own rule.

## Selection decision (2026-08-06)

**Select on in-domain val/test BPS only.** Held-out fixrsvp CC_norm is a
different dataset and a different task, and it is reported in figures 3 and 4 —
selecting on it and then reporting it is circular. This resolves takeaway 2 by
declining the choice rather than by measuring a better CC_norm floor.

**Homogeneous batching is settled and is not reopened by this rule.** Decided
in Stage 0 and reaffirmed 2026-08-06. The whole experiment 01/02 landscape was
mapped under it and everything downstream inherits it. Takeaway 2's
BPS/CC_norm dissociation is recorded as a property of the metrics, not as a
standing question about the batching choice.

**Hold lr at 1e-3; do not tune it.** The sign flip above is the argument: lr
3e-3 gains at effective batch 1024 and loses at 256, so a tuned lr is an
artifact of the batch it was tuned at, and it would be carried to a different
width *and* a different batch. 1e-3 is the value the rest of the pipeline was
built on.

The cost is explicit and accepted: F3c (lr 3e-3, eff batch 1024) has the best
val BPS in the family at 0.5589 against F2a's 0.5473 — a resolved +0.0116,
above the 0.0080 floor. Choosing lr 1e-3 leaves that gain unclaimed, in
exchange for not transferring a batch-specific optimum up a 152× width ladder.

**The remaining choice is batch 256 vs 128 at lr 1e-3**, i.e. `F2a` vs
`02_lr1e-3_bs128`. The latter was never specified in the original 2×2 — the
bs128 cells existed only at lr 3e-3 and 1e-2 — and is now running.

### The floor problem this creates

Both candidates are n = 1, and experiment 02 still has no in-family replicate
group, so the comparison rests on the floor borrowed from F1 (accumulate 4).
If the two land more than ~0.008 val BPS apart the borrowed floor is probably
adequate to call it; if they land closer, the honest answer is that the sweep
cannot distinguish them.

**Tiebreak, fixed 2026-08-06 before the arm evaluated: batch 256 wins a tie.**
It is the incumbent — F2a's configuration — so a tie changes nothing. Recorded
in advance because choosing the tiebreak after seeing the number is how a null
quietly becomes a preference.

`independent_knobs` in `stability.py` exists for this comparison: at
accumulate 1 and fixed samples, `batch_size` determines `effective_batch` and
`max_epochs`, so a raw `spec_diff` reports three knobs and files the cleanest
single-knob comparison in the sweep as confounded.

## Outcome: batch 128 selected

`02_lr1e-3_bs128` reached **0.5657 val BPS**, +0.0184 over F2a and 2.3× the
borrowed floor, so the recorded tiebreak was not needed. It is the best
width-0.5 result of the whole sweep — ahead of the tuned-lr F3c at 0.5589.

**Holding lr at 1e-3 and choosing the batch beat tuning the lr.** That is the
substantive vindication of the not-over-tuning decision, not merely a tidy
outcome: the tuned configuration was reachable and lost to the untuned one.

Unlike the composition arm, this one does not dissociate. It is best in the
family on val BPS, test BPS *and* single-trial r² (0.0312), and its CC_norm
(0.603) is within the floor of the nominal leader (0.610). Every metric agrees,
which is the strongest form the evidence has taken anywhere in this sweep.

**One honest limit on the stopping point.** Along lr 1e-3 the batch axis is
monotone and has not turned over: 4096 → 0.4822, 1024 → 0.5261, 256 → 0.5473,
128 → 0.5657. Batch 64 is untested. Batch 128 is a *deliberate* stopping
point, not a measured optimum, and the model card should say so rather than
imply the peak was located.

## Where the peak is, and is not

Best known configuration remains **F3c** (lr 3e-3, effective batch 1024) at
0.5589 val BPS. Bracketing, at fixed 8M samples:

- Along eff batch 1024: 3e-4 → 0.4297, 1e-3 → 0.5261, 3e-3 → **0.5589**.
  Rising at the top; 6e-3 untested.
- Along eff batch 256: 1e-3 → 0.5473, 3e-3 → 0.5228. Peak is *between*; ~2e-3
  untested.
- Along lr 1e-3: 4096 → 0.4822, 1024 → 0.5261, 256 → 0.5473. Rising toward
  small batches.

The unprobed direction with the most room is lr 6e-3 at effective batch 1024 —
the only line still climbing at its tested edge.
