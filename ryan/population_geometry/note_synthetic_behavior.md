# Note: feeding synthetic (arbitrary) eye-position behavior to the twin is hard

Context: the population-geometry analysis drives the digital twin with in-silico
fixations (Brownian drift over natural patches). We want the model's **behavior**
input to be consistent with the synthetic drift trace ("drift-consistent
behavior"). This turned out to be awkward with the current pipeline. This note
records why, so the **next twin training run** can make synthetic-behavior
datasets a first-class thing.

## What the behavior input actually is

From `experiments/dataset_configs/multi_basic_120_long.yaml` (the fig4 model),
`batch["behavior"]` is the concatenation of two transformed streams of `eyepos`:

- `eye_vel`: `eyepos` → `diff(axis=0)` (finite-diff velocity) → `maxnorm` →
  `symlog` → `temporal_basis` (10 acausal raised-cosine funcs, 50-bin history,
  peak 30–200 ms) → `splitrelu`.
- `eye_pos`: raw `eyepos`, passed through.

Transform implementations: `models/data/transforms.py` (`_make_diff:154`,
`_make_maxnorm:205`, `_make_symlog:199`, `_make_basis:169`, `_make_splitrelu:194`).

## Why it's hard to reproduce for an arbitrary synthetic trace

1. **`maxnorm` has no stored scale.** `maxnorm(x) = x / max(abs(x))`
   (`transforms.py:207`) is computed from the input array itself, at
   dataset-build time, over the whole session's eye trace. There is no persisted
   normalization constant. A synthetic Brownian trace normalized by *its own*
   max lives on a different scale than training, and there is no principled
   session to borrow the scale from. (The physically-meaningful alternative —
   `mul: 240` to get deg/s — is present but **commented out** in the config,
   line 44.)

2. **No standalone `eyepos → behavior` function.** The op chain only runs inside
   dataset construction (transforms applied to the full `DictDataset` array,
   then indexed via `keys_lags.behavior`). There is no inference-time entry point
   that takes an arbitrary `(T, 2)` trace and returns the behavior tensor the
   model expects. Reconstruction means hand-replaying the op chain.

3. **The drive helper zeros behavior.** `ryan/digital-twin-fem/_common.py`
   `compute_rate_map_batched` calls `_zero_behavior(...)` unconditionally
   (`_common.py:197`), so the reusable counterfactual pipeline never threads a
   real/synthetic behavior vector through at all.

4. **The modulator time-averages behavior**, so the effort may not even matter.
   `models/modules/modulator.py:271-273` collapses `(N, T, behavior_dim)` to a
   mean over `T`. The temporally-embedded eye-velocity stream (the whole point of
   the raised-cosine basis) is largely averaged away at read-in.

5. **The twin is near-insensitive to the extraretinal signal.** fig4 found that
   zeroing the behavior leads gives essentially the same predictions
   (see memory `project_fig4_extraretinal_ablation`). So a fragile
   reconstruction buys little for our stimuli.

Net: reconstruction is *possible* (transforms are stateless) but rests on an
unprincipled `maxnorm`-scale imputation, for a signal the model barely uses and
time-averages anyway.

## Recommendations for the next training run

Make "generate a synthetic dataset with arbitrary eye-position behavior" easy:

1. **Persist behavior normalization constants.** Replace per-array `maxnorm`
   with a fit-once-store constant (saved in dataset metadata + checkpoint), or
   switch to a fixed physical scale (`mul: 240` → deg/s, then a fixed divisor).
   Then the identical scale can be reapplied to any synthetic trace.

2. **Expose a standalone `build_behavior(eyepos, config)`** that runs the exact
   op chain on an arbitrary `(T, 2)` trace at inference time, decoupled from
   `DictDataset` build. Use it both in dataset construction and in counterfactual
   drives so they cannot drift apart.

3. **Thread behavior through the drive helpers.** Let
   `compute_rate_map_batched` / `core_forward` accept an optional behavior tensor
   instead of always zeroing it.

4. **Decide, and document, whether behavior should stay time-resolved.** If
   drift *dynamics* are meant to matter, the modulator shouldn't mean-pool
   behavior over `T`; if they aren't, the raised-cosine embedding is wasted
   compute. Either way, make it explicit.

## Decision for now

Per the session decision, we reconstruct drift-consistent behavior by replaying
the `eye_vel` op-chain on the synthetic trace with a `maxnorm` scale imputed from
pooled real fixRSVP traces, concatenated with raw `eye_pos`. This is the
documented workaround until the next training run makes it native.
