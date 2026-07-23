# Next session — hunt more expressive functional forms of `gap = f(saccade, feed-forward drive)`

**Paste the "PROMPT" section below as the first message next session.** Everything
above it is orientation for a human reader.

---

## Why this hunt (motivation)

The perisaccadic model (`../_supp_saccade_model.py`) fits the twin's extraretinal
gap `y = full − ablated` as a saccade-locked **additive + multiplicative** function
of the feed-forward drive:

    y_i(t) = drive_i(t)·Σ_k g_i(t−t_k)  +  Σ_k a_i(t−t_k)        ("both")

Per-neuron ceiling recovers only **~0.403** of the gap on held-out whole trials
(~0.45 in the peri-saccade window). The just-finished **covariate hunt**
(`../covariate_hunt/`, 10 subagents) asked whether the *missing* variance lives in
**other covariates** (drift velocity, eye position, saccade amplitude/direction,
time-since-saccade, drive-gain). Answer: **no** — interpretable covariates add only
**+0.011** and the completeness critic found nothing left; ~58% is not a low-D
function of the marginal signals.

**This hunt flips the question:** maybe we don't need *more covariates* — we need a
**more expressive function** combining the two inputs we already have (**saccade
timing** and **feed-forward drive**). The additive+multiplicative form is rank-1 and
linear-in-drive; the twin's readout is nonlinear and recurrent. Test richer but
still-**interpretable** forms of `f(saccade, drive)`, and — critically — establish a
**black-box ceiling** (any function of those two inputs) as the reference that tells
us how much headroom a better form could possibly capture.

## Ingredients (all already built, reuse them)

- **Target** `y = full − ablated` (gap), per (trial,bin), reliable N=**521**
  (split-half>0.5). Model-vs-model, noise-free.
- **"saccade"** = the saccade-onset lag design `Aadd` (T,B,L), L=37 lags spanning
  −100/+200 ms (microsaccade onsets, `designs[s]["sacc_trial"]/["sacc_bin"]`).
- **"feed-forward drive"** = `drive` = the **ablated (behavior-zeroed) rate**
  `rec["drive"]` per sample. (Also available: drive at other bins for lagged-drive.)
- **Baseline "both"** design = `[1, Aadd(L), drive·Aadd(L)]` → **0.4028** whole-trial
  held-out recovered. This is the number to beat.
- **Harness** = `../covariate_hunt/_supp_saccade_augment.py`:
  `load_augment_context()` (builds `designs`, `reliable_recs`, the reliable mask, and
  the paper baseline), `evaluate_augmentation(col_builders, label, ctx)` (fits
  baseline vs baseline+extra-columns per reliable neuron, **5-fold trial CV seed 0**,
  metric `recovered = 1 − Var_heldout(y−ŷ)/Var(y)`, reports median incremental Δ).
  Any richer form that is **linear-in-parameters** (drive splines, drive², tensor
  products lag×drive, lagged-drive terms) is just augmentation columns — the existing
  harness handles it directly. See `../covariate_hunt/covariate_hunt_log.md` for the
  full protocol and the recurring **paired-median overfit artifact** (Δ-median up
  while aug-median down = not a real gain — always check the aug median rises too).

## What must be BUILT this hunt (two new pieces)

1. **A perisaccade-windowed recovered metric** alongside the whole-trial one, so
   results tie to the "~40% of *perisaccadic* variance" framing. Restrict the held-out
   `recovered` computation to bins within −100/+200 ms of a saccade. Recompute the
   "both" baseline in that window (expected ~0.45) as the in-window number to beat.
2. **A black-box per-neuron ceiling** (the ONLY non-interpretable model allowed, used
   purely as a reference). Per reliable neuron, same 5-fold CV, fit a
   gradient-boosted regressor (or small MLP) with features = [`Aadd` 37-lag design,
   `drive`, a few lagged drives, time-since-saccade]; report median held-out
   recovered. This bounds what ANY `f(saccade, drive)` can reach → the headroom every
   interpretable form is scored against. Build this EARLY (subagent #1–2).

Non-linear-**in-parameters** interpretable forms (e.g. divisive normalization with a
fit κ) need a tiny per-neuron optimizer/grid — the covariate hunt's `_hunt_h7` did a
κ-grid; reuse that pattern.

## Seed hypotheses (functional forms of `f(saccade_lag τ, drive)` — invent more)

1. **Drive nonlinearity in the gain:** replace `drive·g(τ)` with `φ(drive)·g(τ)` for
   interpretable φ — `drive^p`, `log1p(drive)`, saturating `drive/(drive+c)`, `√drive`.
   Does the gain act on a nonlinearly-transformed (saturated/contrast-like) drive?
2. **2D tensor-product surface `g(τ, drive)`:** the modulation-kernel SHAPE depends on
   drive level. Separable "both" is rank-1; try a low-rank (rank-2/3) or spline
   tensor product of (lag τ) × (drive-basis). (Analog of covariate-hunt #9's helpful
   raw-capped speed×drive, but for drive×lag.)
3. **Proper divisive normalization** (McFarland): `full ≈ drive·G/(1+κ·S(τ))` with a
   saccade-driven suppression signal `S(τ)`; `gap = drive·(G/(1+κS) − 1)`. Fit κ per
   neuron (grid/1-D). Compare to the linear additive+multiplicative approximation.
4. **Static output nonlinearity:** `gap = NL(linear saccade+drive) − baseline`
   (softplus / threshold-linear / power) — the readout is nonlinear.
5. **Higher-order saccade×drive:** drive-scaled *additive* offset, `drive²·Aadd`,
   cross terms the rank-1 "both" omits.
6. **Lagged drive:** gap may depend on drive at nearby lags `drive(t−τ)·saccade`, a
   drive temporal profile around the saccade, not just instantaneous drive.

## Orchestration protocol (same as the covariate hunt)

- **New branch** in the VisionCore submodule, e.g. `supp-perisaccadic-form-hunt`.
  Keep ALL work inside `paper/supp_twin_saccade_modulation/perisaccadic_form_hunt/`.
- Copy/adapt the augment harness here (or import from `../covariate_hunt/`); add the
  windowed metric + black-box ceiling. Log every attempt to
  `perisaccadic_form_hunt_log.md` (hypothesis, exact form, median Δ vs "both" AND
  fraction of the black-box gap closed, per-unit notes, verdict).
- **10 subagents, sequential, one hypothesis at a time** (Agent tool,
  general-purpose). Give each: harness location + how to run, the current hypothesis,
  the black-box ceiling, and the full log so far. Build the black-box ceiling first.
- **Promising bar:** ≥ **+0.02** median recovered over "both", OR clearly closes a
  meaningful fraction of the black-box headroom, OR clean reproducible per-unit
  structure. Promising → next subagent refines; **two failures in a row on a form →
  drop it, new independent form.** Stop at 10 or when exhausted.
- **Guardrails:** interpretable only (except the single reference black box);
  held-out numbers only; SAME folds (seed 0) + reliable set (N=521) + metric for every
  comparison; watch the paired-median artifact. Honest negative is a valid outcome —
  if the black box also caps near ~0.45, the perisaccadic gap genuinely is not a
  function of (saccade, drive) alone and the remainder lives in the twin's recurrent
  state (which the covariate hunt already showed marginal signals don't recover).
- At the end: best interpretable form + its recovered, the black-box ceiling, the
  fraction of headroom captured, what remains; update project memory; commit the
  branch (do not push unless asked).

---

## PROMPT (paste as first message)

You are orchestrating an exploratory hunt for a **more expressive but interpretable
functional form** of the digital twin's perisaccadic extraretinal gap
`y = full − ablated` as a function of two inputs it already has: **saccade timing**
and **feed-forward drive** (the ablated rate). The simple additive+multiplicative
("both") model recovers only ~0.403 of the gap (~0.45 peri-saccade window), and the
prior covariate hunt (`../covariate_hunt/`) showed adding *other covariates* barely
helps (+0.011). Hypothesis: the missing variance is in the FUNCTION combining saccade
and drive, not in missing covariates.

Read `paper/supp_twin_saccade_modulation/perisaccadic_form_hunt/NEXT_SESSION_perisaccadic_form.md`
(the orientation above this) and the harness
`paper/supp_twin_saccade_modulation/covariate_hunt/_supp_saccade_augment.py` (plus the
covariate-hunt log for protocol + the paired-median artifact) to load full context
before doing anything.

**First:** create a new branch in the VisionCore submodule (e.g.
`supp-perisaccadic-form-hunt`); keep ALL work inside
`paper/supp_twin_saccade_modulation/perisaccadic_form_hunt/`.

**Build first (subagents #1–2):** (a) a **perisaccade-windowed** held-out `recovered`
metric alongside the whole-trial one (bins within −100/+200 ms of a saccade; recompute
the "both" baseline in-window, ~0.45), and (b) a **black-box per-neuron ceiling** — the
ONLY non-interpretable model allowed, used purely as a reference — a gradient-boosted
regressor (or small MLP) per reliable neuron, SAME 5-fold trial CV (seed 0), features =
[saccade 37-lag design, drive, a few lagged drives, time-since-saccade]; report median
held-out recovered. This bounds what any `f(saccade, drive)` can reach and is the
headroom every interpretable form is judged against.

**Objective.** Beat the "both" baseline on held-out recovered (whole-trial 0.4028 and
in-window ~0.45) using more expressive INTERPRETABLE forms of `f(saccade_lag, drive)`
— drive nonlinearities in the gain, 2D tensor-product/low-rank `g(τ,drive)` surfaces,
proper divisive normalization (fit κ), static output nonlinearities, higher-order
saccade×drive, lagged drive. Black-box function approximators are BANNED as
hypotheses — they are the reference ceiling only. For every interpretable form report
the incremental Δ over "both" AND the fraction of the black-box headroom it closes, on
the SAME folds/reliable set (N=521)/metric.

**Orchestration (you are the orchestrator).** Same protocol as the covariate hunt:
generate one hypothesis at a time; spawn subagents sequentially, up to 10 total
(Agent tool, general-purpose); give each the harness location + how to run, the current
hypothesis, the black-box ceiling, and the full `perisaccadic_form_hunt_log.md` so far.
Each subagent implements the form (interpretable design columns, or a small per-neuron
optimizer for nonlinear-in-params forms), runs the CV harness, and reports median
incremental recovered + diagnostics; grant them latitude to iterate and vet within
their turn. Promising (≥ +0.02 median over "both", or clearly closes black-box headroom,
or clean reproducible per-unit structure) → next subagent refines; two failures in a
row on a form → drop it, new independent form. Log every attempt. Guardrails:
interpretable only (except the one reference black box); held-out only; same
folds/reliable set/metric; watch the paired-median artifact (Δ-median up + aug-median
down = overfit, not gain). Honest negative is valid: if the black box also caps near
~0.45, the perisaccadic gap is genuinely not a function of (saccade, drive) alone and
the rest lives in the twin's recurrent state.

**At the end:** summarize the best interpretable form + its recovered, the black-box
ceiling and fraction of headroom captured, and what remains unexplained; update the
project memory; commit the branch (do not push unless asked).
