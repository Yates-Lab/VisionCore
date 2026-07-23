# Covariate hunt — log

**Objective.** Raise the fraction of the twin's extraretinal ablation gap
`y = full − ablated` recovered on held-out whole trials, above the ~0.40
saccade-kernel baseline, using ONLY interpretable design columns. Honest negative
is a valid outcome.

**Fixed protocol (identical for every hypothesis).**
- Harness: `_supp_saccade_augment.py`. Evaluate via
  `evaluate_augmentation(col_builders, label, ctx)`.
- Target `y = full − ablated` (twin intact − behavior-zeroed), fig2 0.5° frame.
- Reliable set: N = 521 (split-half > 0.5), identical to the baseline paper.
- 5-fold trial CV, seed 0. Metric: median held-out
  `recovered = 1 − Var(y−ŷ)/Var(y)`.
- Baseline design = "both": `[1, Aadd(L), drive·Aadd(L)]`, median base = **0.4028**.
- Every hypothesis = baseline AUGMENTED with interpretable columns; report the
  **incremental** median Δrecovered on the same folds/reliable set.
- **Promising bar:** ≥ **+0.02** median Δrecovered, OR clear reproducible
  per-unit structure. Two subagents in a row failing a line ⇒ drop it, new line.

**Signals available on `designs[session]`** (per trial×bin): `eyepos` (x,y deg,
NaN outside 0.5°), `speed` (drift deg/s), `Aadd` (saccade-onset lags),
`sacc_trial`/`sacc_bin` (microsaccade onsets), `B` (bins/trial; time-in-trial =
`rec["b"]`). Saccade amplitude/direction require extending
`build_saccade_alignment` to carry per-saccade properties (see NEXT_SESSION doc).

**Guardrails.** Interpretable only (no black-box approximators). Held-out numbers
only. Same folds/reliable set/metric for every comparison.

---

## Baseline
- per-neuron "both" ceiling (paper) = **0.4028**, N = 521.
- Harness reproduces 0.4028 exactly with zero extra columns (Δ = 0.0000).
- Trivial demo (time-in-trial linear+quad): Δ median +0.0005 — negligible, as
  expected. Plumbing confirmed.

---

## Attempts

<!-- Each subagent appends one block below. Template:
### N. <hypothesis one-liner>  — [PROMISING | FAIL | PARTIAL]
- **Features (exact):** ...
- **Median Δrecovered:** +0.xxxx  (base 0.4028 → aug 0.xxxx); improved X% of units
- **Diagnostics / per-unit notes:** ...
- **Verdict / next:** ...
-->

### 1. Continuous drift-velocity modulation — PROMISING (script `_hunt_h1_drift_velocity.py`)
- **Features (exact):** per-sample drift speed `designs[s]["speed"][tr,b]` (deg/s,
  central-diff of eyepos), + a "speed-valid" indicator col. Variants: additive
  `[speed, speed²]`; additive `[log1p(speed), log1p(speed)²]`; multiplicative
  `drive·RC₄(log-speed)` (4 raised-cosine bumps); add+mult combos; speed-cap sweep;
  lag sweep (−33..+50 ms).
- **Median Δrecovered:** best single channel **mult drive·logspeed-RC(4) = +0.0079**
  (0.4028→0.4163, 83% units up). Additive speed+speed² = +0.0052 (0.4028→0.4149,
  **88% units up**, sign +461/−60). add+mult logspeed-quad = +0.0065; add+mult RC = +0.0067.
- **Diagnostics / per-unit notes:** speed p50≈low, p99 high, max 54 deg/s (tail =
  small saccades). Effect is instantaneous (best at lag 0; ±8ms already worse).
  Multiplicative (gain-on-drive) ≥ additive. Capping speed≤2 deg/s keeps only
  +0.0023 → the effect uses the *full* drift-speed range, not just slow drift.
  Extremely consistent sign across units (75–88% improve).
- **Verdict / next:** Below +0.02 magnitude but clears the "reproducible per-unit
  structure" bar decisively. REFINE: (a) velocity **direction** (motion-like
  drift-dir×drive; `velocity_dir` was defined but never evaluated), (b) best
  combined add+mult speed+direction model, (c) confirm it stacks with saccade
  kernel (it augments the "both" baseline, so yes by construction) and report the
  best single cumulative model. Fail counter = 0.

### 2. Drift *direction* (fixed-frame velocity) — FAIL for direction; best drift model = speed-only (script `_hunt_h2_drift_direction.py`)
- **Features (exact):** fixed screen-frame eye velocity `[vx, vy]` (central-diff,
  = speed·[cosθ,sinθ]); pure unit direction `[cosθ, sinθ]` (speed-normalized);
  multiplicative `drive·[vx,vy]` and `drive·[cosθ,sinθ]`. Then each stacked on
  top of #1's speed model (`add logspeed-quad + mult drive·logspeed-quad`) to test
  incremental value beyond scalar speed. Base reconfirmed = **0.4028** (Δ=0).
- **Median Δrecovered (direction ALONE, augmenting saccade base):** add velocity
  **+0.0011** (64% up); add unit-dir **+0.0011** (67%); mult drive·velocity
  **+0.0003** (55%, mean −0.0001); mult drive·unit-dir **+0.0014** (64%). All ~7×
  smaller than the speed effect.
- **Does direction add beyond speed? NO.** Speed ref (add+mult logspeed-quad) =
  **+0.0065** (82% up, +425/−96). +add velocity → **+0.0063** (down). +add velocity
  +mult drive·velocity → **+0.0040** (65%, down further). Direction only *dilutes*
  the speed model — pure overfit, never complementary.
- **Diagnostics / per-unit notes:** additive `[vx,vy]` fits per-unit vx/vy betas,
  i.e. it *is* per-unit directional tuning in a fixed frame — and it fails. So
  per-unit drift-direction tuning is effectively already tested (negative). The
  gap modulation is directionally **isotropic**: it depends on drift *speed*, not
  heading. Consistent with a scalar gain/reafference signal, not motion-selectivity.
- **Verdict / next:** Direction = clean NEGATIVE. Best interpretable "drift model"
  for the paper stays speed-only: single best channel = #1's
  `mult drive·logspeed-RC(4)` **+0.0079** (83% up); robust combined
  `add+mult logspeed-quad` **+0.0065** (82% up). The drift-velocity line has
  **plateaued at ~+0.008**, far below the +0.02 bar. Recommend subagent #3 start a
  new INDEPENDENT hypothesis rather than refine drift further. Drift fail counter = 1
  (direction sub-line); overall drift kept as documented small-but-robust effect.

### 3. Eye-POSITION gain field (absolute gaze x,y in 0.5° window) — FAIL (script `_hunt_h3_eye_position.py`)
- **Hypothesis:** ablation zeroes the model's eye-POSITION input, so the gap should
  carry a slow/static modulation vs absolute gaze — a classic gaze gain field —
  distinct from the (isotropic, scalar) drift-speed effect.
- **Features (exact):** instantaneous gaze `designs[s]["eyepos"][tr,b]` and per-trial
  mean gaze. ADD: linear `[x,y]`; quad `[x,y,x²,y²,xy]`; cubic; radial `[r,r²]`;
  trial-mean linear/quad. MULT (gain-field): `drive·[x,y]`, `drive·quad`, `drive·[r,r²]`;
  add+mult combos. Diagnostics: gaze finite for **100%** of valid samples (window
  ±0.5°, r p50=0.23, rmax=0.50) → no zero-fill confound, pos-valid guard returns None.
- **Median Δrecovered (ALONE, on saccade base):** every variant null-to-negative.
  add linear **−0.0001** (47% up, mean +0.0013); add quad −0.0017; add cubic −0.0044;
  add radial −0.0009; trial-mean linear −0.0006. MULT: drive·[x,y] −0.0009 (39% up);
  drive·quad −0.0038; **drive·[r,r²] +0.0006 (58% up)** — the only positive median,
  and add+mult radial **+0.0008 (56% up)** the best overall. Richer polynomials
  monotonically *worsen* (overfit): linear→quad→cubic = −0.0001→−0.0017→−0.0044.
- **Does position ADD beyond drift-speed? NO (redundant/dilutive).** Speed ref
  reconfirmed **+0.0065** (aug median **0.4149**, 82% up). Stacking position on speed:
  every position-augmented **aug median sits BELOW speed-only 0.4149** — add-linear
  0.4135, add-quad 0.4118, mult-quad 0.4077, add-radial 0.4141, trial-mean-quad 0.4131.
  The lone Δ-median bump (speed+add-linear +0.0087 vs +0.0065) is *contradicted by its
  own falling aug median* (0.4149→0.4135) and by every richer position model dropping
  → noise/overfit, not a complementary field.
- **Diagnostics / interpretation:** additive `[x,y]` fits per-unit planar gaze tuning
  and fails; `drive·quad` fits a per-unit 2D gain field and fails. So both the
  classic gain-field and additive-position forms are directly tested (negative). Only
  the isotropic radial term survives at +0.0006–0.0008 — same magnitude/sign story as
  drift-speed (isotropic, scalar), and it does **not** stack on speed, i.e. the tiny
  radial signal is a weak echo of the already-captured speed effect, not new position
  variance. Absolute gaze within the fixation window carries no recoverable gap variance.
- **Verdict / next:** Clean NEGATIVE — eye-position gain field is not a contributor
  (best +0.0008 alone, 25× below the +0.02 bar; redundant with speed). Position line
  **plateaued on first pass**. Recommend subagent #4 start a NEW INDEPENDENT hypothesis
  (e.g. time-since-last-saccade / post-saccadic recovery, or pupil/blink), NOT refine
  position. Position fail counter = 1.

### 4. Saccade amplitude / direction-scaled kernels — FAIL (script `_hunt_h4_amplitude_kernels.py`)
- **Hypothesis:** the baseline "both" design treats every microsaccade identically
  (`Aadd` = 0/1 onset lag indicators). Real saccades vary in amplitude/direction;
  the twin's modulation may SCALE with amplitude (or be direction-tuned). Test
  amplitude-weighted and direction-tuned saccade kernels for incremental value over
  the flat kernel already in X_base.
- **Implementation:** replayed `build_saccade_alignment` exactly (incl. the
  per-trial `get_inds_from_times` ordering) while carrying each mapped saccade's
  original index, then recorded amplitude=`hypot(Δx,Δy)` and dir=`atan2(Δy,Δx)`
  from `saccades.json`. Reconstructed (trial,bin) asserted **equal elementwise** to
  cached `align[s]` for all 25 sessions (gate passed). Cached
  `_hunt_h4_sacc_props.pkl`. Weighted lag design `Aw[tr,b0+lag,li] += w_k` for
  weights: amp raw / sqrt / log1p / mean-centered (each ctr = orthogonal-ish to the
  flat kernel, isolating the incremental amplitude term), and cos θ / sin θ.
  Column builders: full 37-lag, and a parsimonious 4-bump raised-cosine temporal
  projection (37→4 cols). Additive and multiplicative (`drive·Aw`). Base = **0.4028** (Δ=0).
- **Median Δrecovered (ALONE, augmenting saccade base): all NEGATIVE.**
  Full 37-lag kernels overfit badly — add amp_sqrt **−0.0080** (29% up), amp_log −0.0086,
  amp_raw −0.0184, mult drive·amp −0.0421, add+mult amp_ctr −0.0852. Direction 37-lag:
  add cos+sin −0.0134, mult −0.0219. Parsimonious 4-bump RC (the fair low-DOF test):
  **best = add-RC amp_log_ctr −0.0021 (35% up, +182/−339)**; add-RC amp_sqrt_ctr −0.0021;
  mult-RC amp_log_ctr −0.0040; add-RC dir cos+sin −0.0035 (36% up). **No variant, at any
  temporal resolution, beats the flat baseline.** Best overall = −0.0021.
- **Does amplitude add over the flat kernel? NO. Does direction add over amplitude? NO.**
  `add amp_ctr + dir cos/sin` = −0.0438 (worse than amp_ctr alone −0.0176) — direction
  purely dilutes.
- **Diagnostics / interpretation:** amplitude distribution (2184 mapped saccades):
  median **0.55 deg**, p75 0.89, p95 5.6, max 14.8 — a genuine microsaccade core plus a
  large-saccade tail. There IS a real signal: **corr(amplitude, |0–100 ms post-saccade
  gap|) = +0.15** (log-amp +0.16, n=37578 neuron×saccade). But that weak scaling is
  **already absorbed by the flat 0/1 kernel** — adding amplitude DOF only costs held-out
  variance (sign test consistently ~2:1 against; richer parameterizations monotonically
  worse: RC −0.002 → 37-lag −0.008 → add+mult −0.085). The saccade-locked gap is
  amplitude/direction-**invariant** to within what CV can resolve: consistent with a
  stereotyped, event-triggered reafferent transient (fixed shape per onset), not an
  amplitude-graded gain.
- **Verdict / next:** Clean NEGATIVE — amplitude and direction scaling of the saccade
  kernel do not help (best −0.0021, wrong sign, ~10× below the +0.02 bar; direction
  dilutes amplitude). Saccade-kernel-shape line **plateaued on first pass**. Recommend
  subagent #5 start a NEW INDEPENDENT hypothesis (e.g. time-since-last-saccade /
  post-saccadic recovery state, inter-saccade interval, or pupil/blink), NOT refine
  saccade weighting. Saccade-weighting fail counter = 1.

### 5. Eye-velocity TEMPORAL FILTER (integrated velocity-history kernel) — FAIL for temporal integration (script `_hunt_h5_velocity_filter.py`)
- **Hypothesis:** the ConvGRU core integrates continuous eye-velocity over time, so a
  jointly-fit velocity-history kernel should beat #1's INSTANTANEOUS speed (which only
  did per-lag single-shift sweeps, never a joint multi-lag filter). Test additive /
  multiplicative log-speed temporal filters over lag windows, free-lag vs RC-basis, plus
  a 2D (vx,vy) velocity-history filter. Base reconfirmed = **0.4028** (Δ=0).
- **Features (exact):** lagged log1p(speed) matrix `speed[tr, b+lag]` (zero-fill out-of-
  range/NaN + valid indicator) over windows narrow [−50..+25 ms, 10 lags], mid [−100..+33,
  17], wide [−150..+50, 25], past [−200..0, 25]. Free-lag = one col/lag; RC-basis = lag
  axis projected onto 3 or 5 raised-cosine bumps (M @ B). Multiplicative = `drive · filter`.
  2D velocity = lagged vx,vy each RC-projected. Combined = #1 instantaneous
  `[logspeed, logspeed²]` add+mult PLUS the temporal filter.
- **Median Δrecovered (ALONE, on saccade base):**
  free-lag additive log: narrow **+0.0025** (64% up), mid +0.0010, wide +0.0008 — MORE
  lags = WORSE (overfit); raw-speed −0.0003. RC-basis additive log: best wide nb=5
  **+0.0030** (64%), past nb=5 +0.0029, narrow nb=3 +0.0024 (67%). RC-basis
  MULTIPLICATIVE log: narrow nb=5 **+0.0061** (66% up, +344/−177) — best on this line —
  but mid +0.0026, wide +0.0014 (widening overfits). 2D velocity history: null-to-negative
  (best wide nb=3 +0.0007, 53%; others −0.0015..−0.0022) → isotropy confirmed (echoes #2).
- **Does temporal integration beat instantaneous? NO.** #1 instantaneous ref
  (add+mult logspeed-quad) = **+0.0065** (82% up, +425/−96) > best temporal add+mult
  (overfits negative) and > temporal mult narrow +0.0061; #1's single-channel mult
  +0.0079 also unbeaten. **Does history ADD beyond instantaneous? NO — it dilutes:**
  inst + RC-add = +0.0053..+0.0060 (71% up), every one BELOW inst-alone +0.0065 with
  frac-improved dropping 82%→71%; inst + RC add+mult = **−0.0008** (pure overfit). The
  higher raw aug-median of inst+RC-add-wide (0.4163) is a mean/tail artifact — Δ median
  and sign fraction both fall.
- **Free-lag vs RC-basis:** RC-basis wins cleanly (regularization helps): additive
  best +0.0030 (RC) vs +0.0025 (free-lag); multiplicative +0.0061 (RC) reachable only
  after collapsing lags. Richer = overfit is again the failure mode. Best window is the
  NARROWEST one centered on lag 0 — every widening into velocity history costs held-out
  variance.
- **Verdict / next:** Clean NEGATIVE for temporal integration — the velocity effect is
  **instantaneous**, not integrated; a fitted velocity-history kernel neither beats nor
  adds to #1's instantaneous speed (best on-line +0.0061 < #1's +0.0079; stacked on
  instantaneous it only dilutes). Consistent with a moment-by-moment scalar
  drift-speed gain, not a temporally-integrated drive. ~3× below the +0.02 bar. Drift/
  velocity line is now DEFINITIVELY PLATEAUED at #1's instantaneous ~+0.008 across
  four probes (speed #1 +, direction #2 −, position #3 −, temporal-filter #5 −).
  Recommend subagent #6 abandon the velocity family entirely and open a NEW
  INDEPENDENT hypothesis (time-since-last-saccade / post-saccadic recovery state,
  inter-saccade interval, or pupil/blink). Velocity-temporal fail counter = 1
  (drift-family cumulative: 1 promising anchor + 3 negatives).

### 6. Drive-dependent TONIC gain (stimulus-locked, not movement-locked) — PROMISING by structure (2nd positive; script `_hunt_h6_drive_gain.py`)
- **Hypothesis:** the behavior pathway may impose a *tonic* gain on the stimulus
  drive even at zero eye velocity, untied to any saccade/drift event. Then the gap
  is a static function of `rec["drive"]` (ablated stimulus-driven rate) — invisible
  to saccade-triggered averages AND to velocity analyses. Baseline "both" has
  `drive·Aadd` (drive×saccade kernel) but NO standalone drive term outside saccade
  windows, so a tonic drive-proportional gain is unmodeled. `drive` is input-side
  (ablated output, computed w/o behavior), not derived from `full`/`y` — no leakage.
  Base reconfirmed = **0.4028** (Δ=0).
- **DIAGNOSTIC — gap-vs-drive shape (the headline):** within-unit drive deciles,
  mean-centered gap averaged over 521 units: **monotonic DECREASING** —
  `+0.52 +0.24 +0.17 +0.12 +0.10 +0.02 −0.03 −0.12 −0.27 −0.75` (decile 0→9).
  Per-unit corr(decile, centered-gap): **median −0.51, 73% of units negative**.
  Interpretation: the extraretinal/behavior pathway ADDS to low-drive samples and
  SUBTRACTS from high-drive ones = a **divisive-normalization / gain-reduction**
  signature (full < ablated at high drive), roughly linear in drive. drive dist is
  heavy-tailed (p50=28, p99=190, max 670), so RAW polynomials explode.
- **Median Δrecovered (ALONE, on saccade base):** add `drive` (tonic) **+0.0029**
  (82% up, +427/−94); add `log1p(drive)` quad **+0.0035** (82% up) — best static;
  add `drive+drive²` **−0.0086** and add drive-cubic **−0.31** (raw tail explodes);
  per-unit RC(5)/piecewise-linear ≈ +0.001..+0.003 (dilutes vs log-quad).
  `drive·[t,t²]` (drive-scaled trial adaptation) **+0.0046** (81% up) — the tonic
  gain *drifts over the trial*, stronger than the static term.
- **Does drive×speed subsume #1? NO — it IS #1's channel.** `drive·logspeed`
  **+0.0058** (85% up) ≈ #1's multiplicative channel (which already is drive·logspeed);
  `drive·speed` (raw) +0.0024. So H6's velocity interaction merely *reparameterizes*
  #1's mult term — it neither subsumes nor beats it. The genuinely NEW, separable
  H6 effect is the **static/tonic drive gain**, orthogonal to velocity.
- **Does static drive STACK with #1's speed model? YES (separable).** #1 ref
  (add+mult logspeed-quad) = **+0.0065** (0.4149). Stacked:
  `+drive` → **+0.0073** (0.4159); `+log1p(drive)q` → **+0.0082** (0.4162, 82% up);
  `+drive·[t,t²]` → **+0.0094** (0.4177, 82% up, +426/−95) = **BEST cumulative in the
  hunt so far**; `+drive+drive·t` → +0.0091 (mild overfit vs drive·t alone).
- **Best cumulative interpretable model:** `[logspeed, logspeed², drive·logspeed,
  drive·logspeed², drive·t, drive·t²]` (#1 speed + drive×time) → **Δ +0.0094,
  0.4028→0.4177, 82% units improved** (+426/−95). First model to clear +0.009.
- **Verdict / next:** PROMISING by the *reproducible-structure* clause (like #1),
  NOT the magnitude clause — best +0.0094 is still ~2× below the +0.02 bar. But it
  is the **2nd genuine positive** and the first NEW axis since #1: a clean,
  monotonic, 73%-consistent tonic drive-gain (divisive-normalization signature) that
  is **separable from and stacks with** the velocity effect. The `drive·t` win says
  the tonic gain **adapts over the trial**. Recommend #7 pursue the ADAPTATION axis
  (drive × time-since-last-saccade / running-drive average / a proper divisive
  functional form `y ~ −drive/(1+κ·state)` instead of additive log-drive) — this is
  where the drive×t signal points. Honest ceiling read: two gain effects
  (velocity + drive) together give ~+0.0094; the hunt may be plateauing near +0.01
  on interpretable columns. Drive-gain fail counter = 0 (positive line).

### 7. Divisive-normalization functional form + adaptation axis — PARTIAL: confirms structure, does NOT beat #6's additive proxy (script `_hunt_h7_divisive_norm.py`)
- **Hypothesis:** #6 modelled the gain-reduction with ADDITIVE log-drive proxies. Test
  the ACTUAL divisive form. Since `drive ≈ ablated`, a divisive `full = ablated/(1+κ·P)`
  gives `y = full−ablated = −drive·κP/(1+κP)`, so fit single column
  `drive·P/(1+κP)` (OLS absorbs sign/scale; expect NEGATIVE β = gain reduction).
  Sweep κ (small hyperparameter sweep); pool P = (a) instantaneous drive, (b) causal
  exponential running-drive average (τ∈{2,4,8,16} bins, adaptation reading), (c)
  per-trial mean drive. Each divisive column std-scaled per unit (single col → scaling
  is fit-invariant, only fixes conditioning that sank #6's raw drive²). Base = **0.4028** (Δ=0).
- **Features (exact):** `divisive(pool=drive, κ)` = `drive²/(1+κ·drive)` (std-scaled);
  running/trialmean pools analogous; `drive·running(τ)` linear adaptation interaction.
- **Median Δrecovered (ALONE, on saccade base):** divisive pool=drive **κ-sweep is FLAT**
  across κ∈[0,0.2]: +0.0031→+0.0032→…→+0.0025 (**best κ=0.005 → +0.0032, 77% up,
  +403/−118**). Reference additive proxies same folds: `drive` +0.0029, `log1p(drive)q`
  **+0.0035 (82% up)**, `drive·[t,t²]` +0.0046. **Divisive does NOT beat the additive
  proxy** (+0.0032 < +0.0035, and fewer units up: 77% vs 82%) — with std-scaling even
  raw drive² (κ=0) gives +0.0031, so the *shape* of the static drive nonlinearity barely
  matters; any monotone-saturating function caps the static term at ~+0.003.
- **Adaptation axis — the pool is NOT integrated drive (clean negative):** running-drive
  pool monotonically WORSENS with τ (τ=2 +0.0025 → τ=16 +0.0016; frac-up 75%→67%);
  per-trial-mean pool +0.0024; `drive·running-history(τ=4)` +0.0020 → (τ=16) +0.0013.
  ALL below the instantaneous static term AND far below #6's `drive·[t,t²]` +0.0046.
  → the tonic gain drifts with **trial CLOCK-TIME, not accumulated drive**; the
  normalization pool is instantaneous self-drive (τ→0), echoing #1/#5 instantaneity.
  #6's "drive×t" win is genuine trial-phase adaptation, not a drive-integrating pool.
- **Cumulative (best interpretable model):** #1 speed ref = +0.0065 (0.4149). Stacked:
  `+divisive(drive)` → **+0.0083** (0.4161, 83% up) — below #6's `+drive·[t,t²]`
  **+0.0094** (0.4177); `+divisive +drive·[t,t²]` → +0.0092 (divisive is REDUNDANT with
  the drive/log-drive term, adds nothing over #6). **Best cumulative stays #6's
  `[logspeed,logspeed²,drive·logspeed,drive·logspeed²,drive·t,drive·t²]` = +0.0094,
  0.4028→0.4177, 82% units up (+426/−95).**
- **Per-unit sign consistency:** fitting `y ~ [1, divisive(drive,κ=0.005)]` per unit →
  median β **−0.2112, 73% of units NEGATIVE** (gain-reduction sign) — reproduces #6's
  73%-negative decile correlation exactly. Structure is solid; magnitude is capped.
- **Verdict / next:** PARTIAL — the divisive-normalization *interpretation is confirmed*
  (73% negative β, saturating self-drive pool, instantaneous) but the proper functional
  form does **NOT** beat #6's additive log-drive proxy (tied-to-worse alone, redundant
  cumulatively) and the adaptation is trial-clock, not a drive-integrating pool. Drive-gain
  line is now PLATEAUED at #6's **+0.0094** best cumulative across two probes (#6 static+ /
  #7 functional-form ≈). Velocity(#1)+drive(#6) gains together cap the interpretable
  recovery at ~+0.009, ~2× below the +0.02 bar. Recommend #8 abandon functional-form
  refinement and open a NEW INDEPENDENT axis (time-since-last-saccade / post-saccadic
  recovery state, inter-saccade interval, blink/pupil, or a joint velocity×drive gain
  surface), OR accept +0.0094 as the headline interpretable model and stop.
  Drive-gain fail counter = 0 anchor + 1 functional-form null.

### 8. Time-since-last-saccade / post-saccadic recovery state — FAIL (script `_hunt_h8_time_since_saccade.py`)
- **Hypothesis:** the baseline kernel is a FIXED window STA_LAGS = [−100, +200] ms
  (bins −12..+24); beyond +200 ms it predicts nothing. If the twin carries a slow
  post-saccadic gain state that outlasts 200 ms, or a gain that depends CONTINUOUSLY
  on Δt since the last microsaccade, that is unmodeled. Δt is causal/input-side
  (computed from `sacc_trial/sacc_bin` onsets ≤ current bin, per trial), no `full`/`y`.
  Base reconfirmed = **0.4028** (Δ=0).
- **DIAGNOSTIC — gap-vs-Δt shape (the headline; kills the hypothesis up front):**
  pooled mean-centered gap over 521 units, by Δt bin (ms):
  `0–25 −3.1 · 25–50 −6.2 · 50–75 −2.2 · 75–100 +8.4 · 100–150 +6.5 · 150–200 +2.6 ·`
  `200–250 −0.05 · 250–300 −0.7 · 300–400 −1.0 · 400–500 −1.1 · 500–700 −0.9 · >700 −0.6`.
  The entire biphasic transient (suppression → rebound) lives **inside 0–200 ms — exactly
  the fixed kernel window**. Beyond +200 ms the gap is a **FLAT small-negative offset**
  (~−0.7..−1.1), NOT a decaying post-saccadic tail. That flat offset is the tonic
  drive-gain-reduction (#6, full<ablated), not a Δt-dependent recovery. Δt defined for
  ~92% of samples; Δt p50 ≈ 300 ms, max 992 ms — plenty of >200 ms samples to fit, and
  there is simply no slow component there. **No post-saccadic recovery state exists
  beyond the fixed kernel.**
- **Median Δrecovered (ALONE, on saccade base):** additive Δt recovery — every variant
  NULL-to-NEGATIVE: exp(−Δt/τ) bank {100,200,400} **−0.0004** (29% up), single τ 100–500
  −0.0002..−0.0003, RC(Δt) 0–500 ms −0.0003; **beyond-kernel tail (Δt>200 ms) only**:
  exp tail −0.0001 (44% up), RC tail −0.0004 — flat, as the diagnostic predicts.
  MULTIPLICATIVE drive·h(Δt): drive·exp bank **+0.0007** (60% up, aug 0.3993<base),
  drive·RC(Δt) **+0.0010** (63% up, aug 0.4050) — best-alone, but ~20× below the +0.02
  bar and mean −0.0002. Δt×drive-gain axis (log1p(drive)·exp(−Δt/τ)) +0.0008 (59% up).
  ISI: add log1p(ISI) −0.0007; **mult drive·log1p(ISI) −0.0258 (17% up)** = pure overfit.
  Trailing microsaccade rate: add {100,200,400 ms} ≈ 0.0000; mult drive·rate 200 ms
  Δ +0.0000 median (96% up but mean +0.0001, aug 0.4029 — uniform ~0-magnitude, meaningless).
- **Does H8 STACK on the current best model #6? NO.** BEST6 = `[logspeed, logspeed²,
  drive·logspeed, drive·logspeed², drive·t, drive·t²]` reconfirmed **+0.0094** (0.4177,
  82% up, +426/−95). Stacked: `+add exp(Δt) bank` → +0.0089 (0.4175, dilutes);
  `+mult drive·exp(Δt)` → +0.0070 (0.4120, overfits); `+mult drive·rate 200 ms` →
  +0.0094 (0.4178, adds nothing). The lone apparent bump `+log1p(drive)·exp(Δt)` →
  **Δ median +0.0103 but aug median 0.4149 < BEST6's 0.4177** with IDENTICAL mean
  (+0.0090) and sign counts (+425/−96 vs +426/−95) → a paired-median artifact (same
  pattern flagged in #3/#5), NOT a real gain; adding the columns lowered aug recovered.
- **Verdict / next:** Clean NEGATIVE — there is **no post-saccadic recovery state beyond
  the fixed −100/+200 ms kernel**. The saccade-locked biphasic gap is fully contained in
  the existing window; the >200 ms residual is a flat tonic offset already owned by #6's
  drive-gain, not a Δt-decaying transient. ISI and local rate add nothing (or overfit).
  Best-alone +0.0010 (~20× below bar); does not stack on BEST6 (+0.0094 stays). Time-
  since-saccade line **plateaued on first pass**. Best cumulative interpretable model
  remains #6's **+0.0094 (0.4028→0.4177)**. Recommend #9 open a genuinely new axis
  (blink/pupil, or a joint velocity×drive gain SURFACE fit rather than separable terms),
  OR accept +0.0094 as the headline and STOP — three of the last four probes (#5, #7, #8)
  now confirm the interpretable ceiling sits near +0.009. Time-since-saccade fail counter = 1.

### 9. Joint (drift-speed × stimulus-drive) 2D gain surface — PARTIAL POSITIVE: interaction is REAL but is a reparameterization, not a new axis; new headline +0.0108 (script `_hunt_h9_joint_surface.py`)
- **Hypothesis:** #1 (speed gain) and #6 (drive gain) were fit as separable channels;
  test whether a NON-separable 2D function of (log-speed, log-drive) adds held-out
  recovered OVER the separable sum. NOTE the "separable" reference BEST6 already
  contains a bilinear speed×drive term (`drive·logspeed, drive·logspeed²` from #1's
  multiplicative channel), so the genuinely-new quantity is interaction curvature
  BEYOND that bilinear. `speed`/`drive` input-side, no leakage. Base = **0.4028** (Δ=0).
- **DIAGNOSTIC — 2D gap surface (speed decile × drive decile, per-unit-centered, avg
  over 521 units) — there IS genuine interaction:** rows=speed, cols=drive:
  `sp0 +.42 +.32 +.35 +.41 +.41 · sp1 +.45 +.39 +.38 +.40 +.19 · sp2 +.47 +.39 +.43 +.38 +.29 ·`
  `sp3 +.45 +.27 +.24 +.16 +.00 · sp4 −.15 −.79 −1.13 −1.64 −2.98`. The additive-margin
  **residual carries 20% of the surface variance** (interaction SS / total SS = 0.200),
  concentrated in the **extreme high-speed row**: at sp4 the gap collapses steeply with
  drive (high-speed × high-drive = strong suppression −2.98; high-speed × low-drive =
  strong facilitation, residual +0.87). Low/mid speed is ~flat in drive. So the gain
  IS non-separable, but the curvature lives entirely in the fast-drift / microsaccade
  velocity tail.
- **Literal tensor-product interaction — FAIL (overfits, paired-median artifact):**
  RC3×RC3 / RC2×RC2 interaction-only bases stacked on BEST6 → Δ **+0.0077 / +0.0087**
  (down from +0.0094) with aug median **0.4162 / 0.4166 < BEST6's 0.4177** and frac-up
  76–78% < 82% — same overfit failure mode as #3/#5/#8. Bilinear `logdrive·logspeed`
  (+0.0096) and `logdrive·(logsp,logsp²)` (+0.0098) look like Δ-bumps but aug median
  FALLS to 0.4149/0.4142 → artifacts, not real. `drive·logspeed·t` (3-way) −ve (+0.0088).
- **The interaction IS capturable with ONE raw low-DOF term (the positive):** a **raw
  (not log) `drive·speed`** bilinear, with speed **capped ~8–10 deg/s** to exclude the
  large-saccade tail (raw `drive·speed²` explodes: −0.0111, 33% up — the uncapped tail
  is pure noise). Stacked on BEST6: `drive·min(speed,10)` → **+0.0106 (0.4028→0.4182),
  83% up (+430/−91), mean +0.0115** — Δ median AND aug median AND mean AND frac-up ALL
  rise together vs BEST6 (0.4177/+0.0090/82%), so a genuine +0.0012, NOT a paired-median
  artifact. Broad cap plateau: cap 5/8/10/12 → +0.0102/+0.0105/+0.0106/+0.0105 (cap 3
  +0.0090, cap 20 +0.0087 — tail creeps back in). `drive·speed` raw ALONE on saccade
  base = +0.0024 (weak; the win is the tail-curvature it adds to the separable model).
- **Mechanism — it REPLACES #1's log-mult, it is not a new dimension (best & simplest):**
  swapping #1's `drive·(logspeed,logspeed²)` (2 cols) OUT for the single raw-capped
  `drive·min(speed,10)` (1 col) gives **Δ +0.0108 (0.4028→0.4201), 87% up (+452/−69),
  mean +0.0117 — with 6 augmentation columns vs BEST6's 7.** REPLACE cap plateau
  [5,8,10,15] = +0.0101/+0.0108/+0.0108/+0.0103, aug 0.4190–0.4202, uniformly 87% up
  (+451..453/−68..70) → robust, not a cherry-picked hyperparameter. So the "joint
  surface" is NOT genuinely new variance stacked on the separable model — it is a
  **better FUNCTIONAL FORM for the speed×drive gain #1 already modeled**: log-speed
  compresses the fast-drift/microsaccade regime where the interaction curvature lives;
  raw-capped speed weights it correctly. A refinement of #1's mult channel, not a 3rd axis.
- **Best cumulative interpretable model (NEW HEADLINE):**
  `[speed_valid, logspeed, logspeed², drive·min(speed,10 deg/s), drive·t, drive·t²]`
  (6 aug cols on the "both" saccade base) → **Δ +0.0108, 0.4028→0.4201, 87% units up
  (+452/−69).** Supersedes #6's +0.0094 by +0.0014 AND is one column simpler.
- **Verdict / next:** PARTIAL POSITIVE — the literal tensor 2D surface FAILS (overfits,
  same as every "richer" probe), but the diagnostic's real 20%-non-separable curvature
  is captured by a single raw-capped `drive·speed` term that best REPLACES #1's log-mult
  channel: new headline **+0.0108 (0.4028→0.4201, 87% up)**, robust across a broad speed
  cap. This is a reparameterization/refinement of the existing velocity×drive gain, not a
  genuinely new mechanism — the two gain effects (velocity, drive) still cap interpretable
  recovery at ~+0.011, ~2× below the +0.02 bar. Recommend #10 (final): accept **+0.0108**
  as the headline interpretable model and STOP — five straight probes (#5,#7,#8 negatives;
  #9 a form-refinement, not a new axis) confirm the ceiling. Joint-surface fail counter = 0
  (form-refinement positive; literal-tensor sub-line negative).

---

## Summary (subagent #10, FINAL — completeness + synthesis; script `_hunt_h10_completeness.py`)

**HEADLINE interpretable model (exact columns, 6 aug cols on the "both" saccade base):**
```
X_aug = [ speed_valid,            # drift-speed validity indicator
          logspeed, logspeed²,    # additive instantaneous drift-speed gain (#1)
          drive·t, drive·t²,      # tonic drive-gain adaptation over the trial (#6)
          drive·min(speed,10) ]   # raw-capped speed×drive interaction (#9, replaces #1's log-mult)
```
**Reconfirmed exactly:** base **0.4028** → aug **0.4201**, paired Δ-median **+0.0108**
(mean +0.0117), **87% of units up (+452 / −69)**. N = 521 reliable, 5-fold trial CV, seed 0.
Per-neuron recovered: base median 0.4028 IQR [0.284, 0.505] → head median 0.4201 IQR [0.308, 0.521].

**Per-subject / per-session breakdown.**
- **Allen** (N=429): 0.4264 → 0.4385 (Δmed +0.0117, **91% up**).
- **Logan** (N=92): 0.2664 → 0.2680 (Δmed +0.0042, **67% up**) — lower absolute recovery
  and a smaller gain, but still net-positive and majority-up. The effect is not an Allen artifact.
- **Per-session** (21 sessions ≥5 units): median HEAD recovered p25/50/75 = 0.238/0.368/0.449
  (range −0.115…0.607); median Δ p25/50/75 = +0.0028/+0.0089/+0.0160, **90% of sessions positive**.
  Broad and consistent across sessions, not a few-session artifact.

**Completeness critic — the residual `y − ŷ_headline` is NOT low-D recoverable.** Augmenting the
HEADLINE (not the "both" base) with every remaining interpretable signal and reading the
INCREMENTAL Δ over the headline (guarding the paired-median artifact: Δ-median up **and** aug-median
down = overfit, not gain):

| candidate signal | incr Δmed | aug-median 0.4201→ | % units up | verdict |
|---|---|---|---|---|
| (a) eye position `[x,y,x²,y²]`+valid | −0.0011 | 0.4168 | 39% | negative |
| (b) time-since-last-sacc exp bank {100,200,400 ms} | −0.0001 | 0.4194 | 45% | null-neg |
| (c) extended saccade kernel (lags outside −100/+200 ms, add) | −0.0011 | 0.4181 | 12% | strong neg |
| (c) extended saccade kernel (add+mult) | −0.0034 | 0.4144 | 10% | overfit |
| (d) time-in-trial `[t,t²]` standalone | −0.0004 | 0.4191 | 39% | null-neg |
| (e) local microsaccade rate (add) | +0.0000 | 0.4201 | 54% | ARTIFACT (aug flat) |
| (e) local microsaccade rate (add+mult) | +0.0000 | 0.4201 | 58% | ARTIFACT (aug flat) |
| (f) log-drive higher orders `[ld, ld², ld³]` | −0.0003 | 0.4196 | 47% | null-neg |
| **ALL candidates (kitchen sink)** | **−0.0057** | **0.4089** | 26% | **hurts** |

**Nothing is left on the table.** Every candidate is null-to-negative or a flagged paired-median
artifact; the kitchen-sink of all signals at once *lowers* aug recovered by −0.0112. The two
apparent Δ=+0.0000 microsaccade-rate rows are artifacts (aug-median does not move). Note (f):
the headline carries no *pure* static-drive term, yet higher-order log-drive still fails to add —
`drive·t` already absorbs the tonic drive-gain. The residual carries no recoverable low-dimensional
marginal structure — consistent with the clean negatives across #2–#8 (direction, position,
amplitude/direction kernels, temporal integration, divisive functional form, post-saccadic state).

**Where the +0.0108 concentrates (broad, drive-weighted).** Per-unit Δ: p10 −0.0025 · p25 +0.0047
· p50 +0.0108 · p75 +0.0194 · p90 +0.0299; 87% of units improve; the top-10% units hold 38% of the
total positive-sum Δ (a heavy positive tail, but the median unit still clearly gains → broad, not a
few-unit spike). High-Δ units are **high-drive**, not high-drift-speed: corr(Δ, mean drive) **+0.157**
(top-Δ quartile mean drive 37.5 vs bottom 18.5), corr(Δ, drift-speed variance) **−0.178**,
corr(Δ, gap variance) −0.03. The gain terms (`drive·t`, `drive·min(speed,10)`) scale with firing
magnitude, so strongly-driven units gain the most absolute recovered variance — as expected for a
drive-scaled gain, not a velocity-tuned one.

**HONEST CEILING.**
- Saccade kernel alone recovers **~40.3%** of the twin's extraretinal ablation-gap variance (median unit).
- The interpretable gain terms (drift-speed gain + tonic/adaptive drive gain + their raw-capped
  interaction) add the median unit to **~42.0%** (aug-median; paired Δ-median +1.08 pts) — a small,
  robust, 87%-consistent improvement, **but ~2× below the +0.02 promising bar.**
- **~58% of the gap variance remains unrecovered by any simple marginal function** of the available
  interpretable signals (velocity, position, drive, saccade timing/amplitude/rate, trial phase,
  post-saccadic state). Ten hypotheses, ~90 evaluated variants: the residual is distributed across
  the twin's **nonlinear recurrent computation** and is not reducible to a low-dimensional
  interpretable design. **Accept +0.0108 (0.4028→0.4201) as the final headline and stop.**
