# Next session — hunt the remaining ~60% of the twin's extraretinal gap

**Paste the "PROMPT" section below as the first message next session.** Everything
above it is orientation for a human reader.

---

## What exists in this folder (built last session)

Goal so far: understand the twin's **extraretinal contribution**, isolated as
`gap = full − ablated` where `ablated` = behavior(eye-vel/eye-pos)-input zeroed.
All on **fig2's 0.5° fixation frame** (reused cache, no new inference).

Files (`VisionCore/paper/supp_twin_saccade_modulation/`):
- `_supp_saccade_data.py` — reuses `outputs/cache/supp_twin_fig2frame_conditions.pkl`
  (25 sessions; per trial×bin: `robs`, `rhat_used[{intact,zeroed,stabilized}]`,
  `dfs`, `eyepos`, `neuron_mask`; all affine-rescaled). Maps each session's
  `saccades/saccades.json` onto cached trials via `get_inds_from_times`,
  **bit-wise gated** on reconstructed vs cached `eyepos` (→
  `supp_saccade_alignment.pkl`: per session `sacc_trial`, `sacc_bin`). Builds the
  saccade-triggered-average bundle (`supp_saccade_sta.pkl`) incl. per-neuron
  split-half `reliability`. Constants: `DT=1/120`, STA window −100/+200 ms
  (`STA_PRE=12`, `STA_POST=24`, `STA_LAGS` len 37), `RELIABLE_THRESHOLD=0.5`,
  example = `Allen_2022-04-08` n62.
- `generate_supp_twin_saccade_modulation.py` — descriptive figure: the gap is a
  biphasic, microsaccade-locked STA, consistent across units.
- `_supp_saccade_model.py` — the **evaluation harness**. Fits a saccade-locked
  gain+offset model to `gap`:
  `y(t) = drive(t)·Σₖ g(t−tₖ) + Σₖ a(t−tₖ)` (drive=ablated rate; linear
  superposition over that trial's microsaccades). Model ladder
  `{add, mult, both} × {per-neuron free kernels, pooled rank-1 (shared g₀/a₀ +
  per-neuron wᵍ/wᵃ, ALS), pooled+δ latency}`. **Metric = fraction of the ablation
  gap recovered on held-out WHOLE trials**, `recovered = 1 − Var(y−ŷ)/Var(y)`,
  **5-fold CV over trials within session** (folds from `rng(0)` in
  `_build_designs`), reliable units (n=521). Key funcs: `_build_designs`,
  `_neuron_records`, `_design_cols`, `_fit_per_neuron`, `_als`, `_fit_pooled_cv`,
  `_ols`, `_recovered`, `compute_model_bundle`, `print_model_stats`. Cache
  `supp_saccade_model.pkl`.
- `generate_supp_saccade_model.py` — 6-panel modeling figure.

**Findings.** Saccade-kernel model recovers **~0.40** of the gap (per-neuron
ceiling; pooled+δ 0.36 = 90% of ceiling → the locked part is stereotyped). Both
additive AND multiplicative (add 0.33 / mult 0.36 / both 0.40). Shared gain ≈ 30%
divisive suppression @ ~40 ms + additive +5.5 sp/s @ ~70 ms (McFarland-like). ⇒
**~60% of the ablation gap is NOT microsaccade-locked** — the target for next
session.

**Signals available for new hypotheses** (all per trial×bin unless noted):
- `eyepos` (within 0.5°): absolute gaze (x,y); derived **drift velocity/speed**
  (finite diff); distance from center. NOTE: bins outside 0.5° are NaN
  (fixation gate) → position/velocity hypotheses are limited to the fixational
  range; a wider range would need the un-built 1.0° inference.
- `saccades.json` per session (`YatesV1Session(name).sess_dir/'saccades'/
  'saccades.json'`): `start_x/y`, `end_x/y` → **amplitude, direction**;
  `start_time`, `duration`. The current kernel treats every microsaccade
  identically — amplitude/direction is unused. To use it you must extend
  `build_saccade_alignment` to carry per-saccade properties aligned to the same
  (trial,bin) list.
- time-in-trial (bin), time-since-last-saccade (continuous), inter-saccade
  interval, microsaccade rate.
- `robs`/`rhat` exist but the **target is the twin gap** (model-vs-model,
  noise-free) — keep it that way.

**Gotchas:** per-condition affine rescaling → keep the per-neuron intercept
(harness already has it). Use held-out CV recovered, never in-sample. Keep the
SAME folds (seed 0), reliable set, and metric across every comparison.

---

## PROMPT (paste as first message)

You are orchestrating an exploratory search for the ~60% of the digital twin's
extraretinal contribution that a simple microsaccade-locked model does NOT
explain. Read `VisionCore/paper/supp_twin_saccade_modulation/
NEXT_SESSION_variance_hunt.md` (the section above this) and the four scripts it
describes to load full context before doing anything.

**First: create a new branch** in the VisionCore submodule (this is entirely
exploratory), e.g. `supp-variance-hunt`. Keep ALL work inside
`paper/supp_twin_saccade_modulation/`.

**Objective.** Increase the fraction of the ablation gap (`full − ablated`)
recovered on held-out whole trials, above the ~0.40 saccade-kernel baseline,
using ONLY simple, interpretable functions of the available signals (gaze
position, drift velocity, saccade amplitude/direction/timing, time-in-trial,
etc.). Black-box function approximators are banned — the point is *understanding*
which interpretable signals carry the rest. It is possible no simple function
exists; an honest negative is a valid outcome.

**Harness.** Reuse/extend `_supp_saccade_model.py`. Add a general evaluator:
given a per-(trial,bin) design-matrix builder, fit `y = X·β` per reliable neuron
with the EXISTING 5-fold trial CV (seed 0) and report median held-out
`recovered`. Every hypothesis = the saccade "both" baseline design AUGMENTED with
new interpretable columns; always report the incremental recovered over baseline,
on the same folds/reliable set/metric. Log every attempt to
`variance_hunt_log.md` (hypothesis, exact features, median Δrecovered, per-unit
notes, verdict).

**Orchestration protocol (you are the orchestrator):**
- Generate **one hypothesis at a time**. Do not enumerate all upfront.
- Spawn subagents **sequentially, one after another, up to 10 total** this
  session (Agent tool, general-purpose). Give each subagent: the harness location
  and how to run it, the current hypothesis, and the full `variance_hunt_log.md`
  so far.
- Each subagent implements the hypothesis as interpretable design columns, runs
  the CV harness, and reports median incremental recovered + diagnostics. Grant
  them latitude to be **creative and tenacious**: if they notice the effect
  concentrates (specific units, a better transform, a nonlinearity), they should
  iterate and vet thoroughly WITHIN their turn before concluding, and say so.
- **Promising** (set a concrete bar, e.g. ≥ +0.02 median recovered, or clear
  reproducible per-unit structure) → the **next subagent follows up** to refine
  that direction; reset the fail counter on any real improvement.
- **Two-strike rule:** once **two subagents in a row fail to improve** a
  hypothesis line, drop it and generate a NEW INDEPENDENT hypothesis.
- Stop at 10 subagents or when ideas are exhausted.

**Seed hypotheses (you choose one at a time; invent your own too):**
1. Saccade amplitude/direction-dependent kernels (scale by amplitude; split or
   cosine-tune by direction) — likely raises the *locked* fraction beyond 0.40.
2. Continuous drift-velocity modulation: interpretable gain+offset as a function
   of instantaneous eye speed, not just discrete saccades.
3. Eye-position gain field: gap vs absolute gaze (x,y), low-order/2D-smooth.
4. Time-since-last-saccade continuous recovery/adaptation beyond the fixed window.
5. Drift-direction × drive interaction (motion-like modulation).

**Guardrails.** Interpretable only. Same CV folds + reliable set + metric for
every comparison. Held-out numbers only. Report negatives honestly. At the end:
summarize the best interpretable model, total recovered achieved, and what
remains unexplained; update the project memory; commit the branch (do not push
unless asked).
