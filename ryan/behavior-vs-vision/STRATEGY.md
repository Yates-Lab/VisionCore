# Behavior-vs-Vision: Strategy and Analysis Roadmap

**Owner:** Ryan Ressmeyer
**Project:** V1 fovea fixational eye movement (FEM) manuscript
**Manuscript:** `fem-v1-fovea/main.tex`
**Last updated:** 2026-04-28

This document is the canonical strategy memo for the "behavior vs vision-only digital twin" thread of the manuscript. It is the context to load at the start of any future session that will run analyses for this thread.

---

## 0. tl;dr

We have two image-computable digital twin models of foveal V1:

- **Behavior model** — `learned_resnet_concat_convgru_gaussian` — gets shift-corrected stimulus *plus* a `behavior` tensor (eye velocity history + eye position).
- **Vision-only model** — `learned_resnet_none_convgru_gaussian` — gets only the shift-corrected stimulus.

Both models are trained on the same data. On held-out **fixRSVP** trials, the behavior model has consistently higher BPS (median ΔBPS ≈ 0.05 across both subjects, Wilcoxon p ≪ 0.001 per subject; see `compare_models_fixrsvp.py`).

The question this document is built around:

> **What is the additional variance captured by the behavior model telling us about the visual code in the fixRSVP task?**

The first analysis we tried — Σ_Δ subspace alignment (`subspace_residual.py`) — produced a structurally suggestive result (low-rank, non-Int-aligned residual; FEM-leaning leading direction in Allen) but was not compelling in isolation: the effect at k=5 is modest, the trial-shuffle null I built tested the wrong thing (subspace alignment is invariant to trial-pattern), and the analysis is descriptive rather than mechanistic.

This memo reframes the question, lays out a layered breakdown, and prescribes a ranked set of analyses that — if they work as predicted — turn this into a Fig 4 that bridges the existing Fig 2 (empirical FEM dominance) and Fig 5 (FEM sharpens info) into a single image-computable account.

---

## 1. Background and current state

### 1.1 Where the manuscript stands

The current `main.tex` builds the following arc:

| Figure | Result | Implementation status |
|---|---|---|
| Fig 1 | Population recordings during foveal fixation; eye-conditioned PSTH sharpening; population eye-state decoding | partial (see `ryan/fig1`) |
| Fig 2 | Law-of-total-cov decomposition: Σ_total = Σ_PSTH + Σ_FEM + Σ_int. FEMs dominate trial-to-trial *rate* variability; Σ_FEM aligned with Σ_PSTH; Fano + noise correlations collapse after FEM correction | mature; cache at `outputs/cache/fig2_decomposition.pkl`, fig generator at `ryan/fig2/generate_figure2.py` |
| Fig 3 | Image-computable model with shifted stimulus reproduces tuning + free-viewing variance | mature; cache at `outputs/cache/fig3_digitaltwin.pkl`, scripts at `ryan/fig3/` |
| Fig 4 | TBD — this memo's target | not built |
| Fig 5 | FEMs sharpen spatial information; optimized eye traces match real FEMs; acuity-task Fisher info | partial |

The **central empirical claim** of the paper, established in Fig 2, is that what looks like internal noise is largely *FEM-driven structured rate modulation*. The **central computational claim**, building toward Fig 5, is that this modulation is functional — FEMs serve vision, not corrupt it.

### 1.2 The recent finding

We compared two digital twin variants (architecturally identical except for the behavior modulator) on **fixRSVP** — a fixation condition with rapidly flashed natural images. See:

- `VisionCore/ryan/behavior-vs-vision/compare_models.py` — validation BPS + perisaccadic comparison on free-viewing classes.
- `VisionCore/ryan/behavior-vs-vision/compare_models_fixrsvp.py` — fixRSVP-specific, with affine rescaling of rhat per cell, ccnorm, single-trial r².

Both models receive **shift-corrected stimuli** (the eye-trace is baked into the stimulus by the Yates 2023 shifter at dataset packaging time; `data-yates-v1/DataYatesV1/utils/post_shifter_processing.py:13-44`). The behavior tensor adds:

- `eye_vel`: 10 raised-cosine basis functions over a 50-bin acausal history of eye velocity, splitrelu'd → 20 channels.
- `eye_pos`: raw 2D eye position → 2 channels.

(Defined in `experiments/dataset_configs/multi_basic_120.yaml:25-50`. Concat-ed into the core after the convnet via `models/modules/modulator.py:47-115`.)

Result: **moderate but consistent ΔBPS in favor of the behavior model on fixRSVP**, with the gap visible in both Allen and Logan subjects. Saved figures: `outputs/figures/behavior-vs-vision/fig1_llr_histograms.{pdf,png}`, perisaccadic figures, etc. Per-session metrics cache: `outputs/cache/behavior_vs_vision_fixrsvp.pkl`.

This is in tension with the strict reading of Figs 2-3, which would have the *retinal* input fully account for FEM-driven variability. If retinal motion (delivered via the shifted stim) were the whole story, behavior regressors should add nothing.

### 1.3 The pilot analysis (what didn't land)

`subspace_residual.py` computed:

- Σ_Δ = Cov(rhat_beh − rhat_vis) per session, pooled across trials × time on fixRSVP.
- Subspace alignment of Σ_Δ vs Σ_PSTH, Σ_FEM, Σ_int (the latter from the Fig 2 cache).
- Trial-shuffled behavior-input null.

Results across 24 sessions (cache: `outputs/cache/behavior_vs_vision_residual_subspace.pkl`):

- Σ_Δ is consistently **low-dimensional**: PR ≈ 3.2 (cf. PR(Σ_FEM) ≈ 3.2, PR(Σ_int) ≈ 42).
- Σ_Δ **avoids the internal-noise subspace**: pooled k=5 capture is 0.73 in Σ_PSTH, 0.70 in Σ_FEM, 0.48 in Σ_int (Wilcoxon PSTH > Int p = 3 × 10⁻⁷).
- In Allen sessions (n=11), the **leading direction of Σ_Δ aligns ~0.91 with the leading FEM direction but only ~0.54 with the leading PSTH direction** (Wilcoxon p = 5 × 10⁻⁴ for the asymmetry). Logan (n=13) does not show this asymmetry — its PSTH and FEM directions are themselves entangled in small populations.

**Why this is not a compelling result on its own:**

1. The k=5 capture difference between PSTH/FEM (0.7) and Int (0.48) is real but quantitatively modest, and at k=5 PSTH and FEM are nearly indistinguishable (Σ_FEM is itself PSTH-aligned per Fig 2).
2. The leading-mode asymmetry exists only in Allen and only at k=1, where eigenvector estimates are sensitive to noise.
3. The trial-shuffle null is the *wrong control*: subspace overlap is invariant to the trial-by-trial pattern within a subspace, so shuffling behavior across trials still yields the same overlap. The pilot shuffle null tested whether the *trial-specific* behavior pattern matters; the subspace claim concerns where in population space the behavior model adds capacity, regardless of trial pattern.
4. Most fundamentally: the analysis describes *where* the residual lives; it does not address *what the residual represents*.

---

## 2. The question, decomposed

> **What is the additional variance captured by the behavior model telling us about the visual code in the fixRSVP task?**

This is one question, but it has three nested layers, and each requires different evidence:

### Layer 1 — Input-level

> *What information is the behavior input delivering that the shifted stimulus alone cannot?*

Possibilities:
- Residual shifter error (the shift correction is imperfect; behavior cleans it up).
- Eye-position-dependent gain (gaze-dependent multiplicative modulation of visual responses).
- Peri-saccadic / motor-related modulation.
- Microsaccade-related transients.
- Arousal proxies correlated with eye-movement statistics.

The Σ_Δ subspace analysis touches Layer 1 — it tells us the residual lives in FEM-aligned dimensions. But it does not separate retinal-residual from extraretinal.

### Layer 2 — Mechanism

> *What internal representation does the model build using that information?*

Possibilities:
- Multiplicative gain on stimulus features (gain-field-style)
- Additive offset that depends only on behavior (drift baseline)
- Stimulus-conditional modulation (rate response-shape changes with eye state, not just amplitude)
- Channel-specific use of eye_pos vs eye_vel

Channel ablations and per-cell dependency maps land here.

### Layer 3 — Function

> *What does that representation do for the V1 code? Is the eye-conditioned modulation functionally meaningful for vision, or just descriptive?*

This is the layer that lifts the manuscript to a Nature/Neuron-tier story. The function question splits further:

- Does the behavior modulation reproduce the empirical **shared variability** documented in Fig 2 (the noise-correlation collapse story)?
- Is the modulation **information-limiting** (parallel to tuning gradients, hurting decoding) or **information-preserving** (orthogonal, leaving decoding intact, possibly enhancing it)?
- Does the behavior-modulated representation match Fig 5's "FEMs sharpen spatial info" claim at the population level?

The Σ_Δ pilot does not reach Layer 3.

### Why we have to reach Layer 3

The current Fig 4 sketch in the manuscript ("the behavior model captures more variance than vision-only") is descriptive. Reviewers at high-impact venues will treat it as suggestive, not conclusive. We need at least one analysis that demonstrates the behavior modulation is *the* image-computable account of a phenomenon documented elsewhere in the paper, or *causally relevant* to V1's coding capacity.

---

## 3. The reframing — what makes this compelling

The cleanest framing I can find:

> **The behavior model is the first image-computable model to recover the FEM-driven shared variability documented in Figure 2.**

Why this is a tight claim:

### 3.1 The math

Empirical Σ_FEM (Fig 2 / `main.tex` Methods §"Decomposition of Neural Variability via the Law of Total Covariance"):

$$\Sigma_{\mathrm{FEM}} \equiv \mathbb{E}_t\!\left[\mathrm{Cov}_e\!\left(R(t,e) \mid t\right)\right]$$

That is, fix a stimulus time $t$; look at the conditional rate $R(t,e) = \mathbb{E}_i[Y \mid t, e]$ as a function of eye state $e$; take its covariance across $e$ at fixed $t$; average across $t$.

For a deterministic image-computable model with output $\hat r(t, e)$, the analogous estimator is:

$$\hat\Sigma_{\mathrm{FEM}}^{\text{model}} = \frac{1}{T} \sum_{t=1}^T \mathrm{Cov}_i\!\left(\hat r(t, e_i)\right)$$

— at each stim time $t$, take the across-trial covariance of the model's predictions (since each trial $i$ has its own eye trajectory $e_i$). Average across stim times.

This is *exactly the same theoretical object* as the empirical Σ_FEM, just estimated from model outputs instead of spike counts.

### 3.2 What each model is doing

In fixRSVP, the world-frame stimulus is repeated across trials; the eye trace is trial-specific.

- **Vision-only model** sees the shift-corrected (retinal-frame) stimulus. Because the shifter uses the per-trial eye trace to put the image in retinal coordinates, the retinal-frame stimulus *is* trial-specific — it carries FEM information *through the input*. Vision-only can therefore predict nonzero across-trial variance. But its access to FEM is mediated entirely by what the convnet can extract from the retinal-frame image sequence.
- **Behavior model** also sees the shift-corrected stimulus *and* receives eye_pos / eye_vel as a separate input channel. It can therefore compensate for shifter error and represent extraretinal modulation that the convnet alone cannot derive.

So the comparison is:

| Quantity | What it measures |
|---|---|
| $\hat\Sigma_{\mathrm{FEM}}^{\text{vis}}$ | FEM-driven variance recoverable from the shift-corrected retinal input alone |
| $\hat\Sigma_{\mathrm{FEM}}^{\text{beh}}$ | FEM-driven variance recoverable when behavior is provided as an explicit input |
| $\Sigma_{\mathrm{FEM}}^{\text{empirical}}$ (Fig 2) | Total empirical FEM-driven variance |

If $\hat\Sigma_{\mathrm{FEM}}^{\text{beh}} \approx \Sigma_{\mathrm{FEM}}^{\text{empirical}} \gg \hat\Sigma_{\mathrm{FEM}}^{\text{vis}}$ in magnitude, alignment, and pairwise correlations, then:

- Fig 2's empirical FEM dominance is **mechanistically reproduced** by an image-computable model.
- The vision-only baseline **fails to reproduce it** despite having access to the retinal-frame stimulus → the behavior input carries non-redundant information about the FEM-driven structure.
- The "extraretinal" claim has a concrete, testable form: it is whatever fraction of $\Sigma_{\mathrm{FEM}}^{\text{empirical}}$ the behavior model recovers but vision-only does not.

### 3.3 The chain Fig 2 → Fig 4 → Fig 5

- **Fig 2** (empirical): FEMs dominate shared variability.
- **Fig 4** (image-computable): The behavior model recovers that shared variability; the vision-only model does not.
- **Fig 5** (function): The recovered FEM channel sharpens spatial information at the population level (extension of the existing Fig 5 sketch).

This is a tight three-figure chain that spans empirical claim → mechanistic model → functional consequence. None of the existing manuscript figures do this on their own; together they would.

---

## 4. Analyses

Ranked by my read of how much each contributes to the chain in §3.3. Run in order; A1 is the gating test for the rest.

### Tier 1 — Image-computable recovery of empirical Σ_FEM

#### A1. Σ_FEM recovery test

**Question addressed:** Does the behavior model reproduce the empirical Σ_FEM from Fig 2 in magnitude, direction, and pairwise correlation structure? Does the vision-only model fail to?

**Inputs:**
- `rhat_beh`, `rhat_vis` per session: shape `(NT_trials, T_stim_bins, NC_neurons)` from the behavior and vision-only models on fixRSVP. Currently *not* cached by `compare_models_fixrsvp.py` (only per-cell metrics are). Either (a) extend `compare_models_fixrsvp.py` to cache the rhat tensors, or (b) extend `subspace_residual.py` (which already runs inference) to save them.
- Empirical Σ_FEM, Σ_PSTH, Σ_int from `outputs/cache/fig2_decomposition.pkl`. Each session has `mats[w_idx]['PSTH']`, `mats[w_idx]['Intercept']` (= Σ_rate), and `mats[w_idx]['Total']`, indexed by counting window. Use `w_idx = 0` (≈ 8.33 ms, matches the model's 1/120 s timestep).
- Robs and dfs aligned on the same fixRSVP trials.
- Affine-rescale rhat to match observed spike scale per cell (same protocol as `compare_models_fixrsvp.py:_compute_metrics`, using `rescale_rhat` from `eval.eval_stack_utils`).

**Algorithm:**

```
For each session:
    Restrict to neurons in the intersection of fig2's neuron_mask and the
    fixrsvp neuron mask (same intersection as subspace_residual.py).

    Affine-rescale rhat_beh and rhat_vis per neuron.

    For each stim time t (psth_inds value):
        i_valid = trials with valid samples at time t (dfs > 0)
        if len(i_valid) < min_trials_per_t: continue
        beh_centered = rhat_beh[i_valid, t, :] - rhat_beh[i_valid, t, :].mean(0)
        vis_centered = rhat_vis[i_valid, t, :] - rhat_vis[i_valid, t, :].mean(0)
        Sigma_pred_beh_t = beh_centered.T @ beh_centered / (len(i_valid) - 1)
        Sigma_pred_vis_t = vis_centered.T @ vis_centered / (len(i_valid) - 1)

    Sigma_pred_beh = E_t[Sigma_pred_beh_t]   # weighted by len(i_valid)
    Sigma_pred_vis = E_t[Sigma_pred_vis_t]

    Sigma_FEM_emp = fig2 cache for this session, intersected to common cells.
```

**Metrics, per session:**
1. **Magnitude ratio**: $\|\hat\Sigma_{\mathrm{FEM}}^{\text{beh}}\|_F / \|\Sigma_{\mathrm{FEM}}^{\text{emp}}\|_F$ and same for vis. *Expected*: beh approaches 1, vis substantially smaller.
2. **Trace ratio**: $\mathrm{tr}(\hat\Sigma_{\mathrm{FEM}}^{\text{beh}}) / \mathrm{tr}(\Sigma_{\mathrm{FEM}}^{\text{emp}})$. Cleaner than Frobenius if we're focused on total variance.
3. **Eigenvector overlap** between $\hat\Sigma_{\mathrm{FEM}}^{\text{beh}}$ and $\Sigma_{\mathrm{FEM}}^{\text{emp}}$ — reuse `VisionCore.subspace.symmetric_subspace_overlap` at k=1, k=5.
4. **Pairwise noise correlation scatter**: ρ_pred^beh vs ρ_empirical (off-diagonal of the correlation matrices). Same for vis. Expected: beh on the diagonal, vis flat near 0.
5. **Pairwise noise correlation distribution**: histogram of ρ values for empirical, beh, vis. Expected: empirical and beh have matched skew/spread, vis is concentrated near 0.
6. **Per-cell variance ratio**: diag($\hat\Sigma_{\mathrm{FEM}}^{\text{beh}}$) / diag($\Sigma_{\mathrm{FEM}}^{\text{emp}}$). Per-cell scatter, color-coded by subject.

**Decision tree on outcomes:**

| Outcome | Interpretation | Manuscript implication |
|---|---|---|
| beh ≈ empirical >> vis ≈ 0 | Behavior model fully recovers FEM; vision-only fails because shifted stim doesn't carry enough FEM info | Strong "image-computable + behavior demonstrates Fig 2" story → Fig 4 lands |
| beh ≈ empirical >> vis > 0 | Both capture some FEM; behavior captures all of it | Story still works; vision-only partial recovery is not surprising |
| beh > vis but neither ≈ empirical | Neither model captures all FEM-driven structure → suggests pupil / arousal / unmeasured behavior | Useful but weaker story; Fig 4 becomes "a portion of FEM is recoverable from eye state alone" |
| beh ≈ vis | Behavior input adds no across-trial structure → contradicts ΔBPS finding → debug pipeline | Stop and investigate |
| vis ≈ 0 | Shifter is essentially perfect at compensating for FEM in retinal stim → FEM-driven empirical variance is *necessarily* extraretinal in origin | Even cleaner extraretinal story |

**Implementation notes:**
- Pool across stim times *only after* per-time covariance is computed; do not concatenate across times for a single covariance calculation (that would mix in PSTH variance, exactly what we want to remove).
- Weight per-time covariances by the number of valid trials, not equally.
- Expect `Sigma_pred_vis_t` to be small but not zero (shifter is imperfect, model is nonlinear); it tells us how much of the shifter-residual contributes.
- Watch for sample-size bias: per-time covariances are estimated from `n_trials` ≈ 50–150 samples per session in C-dimensional space (C up to ~150 neurons). Use at least 10 trials per stim time.

**Suggested figure layout (cross-session summary):**
1. Magnitude ratio (beh, vis) per session, paired bar plot, Allen vs Logan.
2. Eigenvector overlap (Sigma_pred^beh vs empirical Σ_FEM) per session, scatter vs n_cells.
3. Pairwise noise correlation scatter: ρ_empirical vs ρ_pred per pair, panels for beh and vis side by side.
4. Per-pair histogram: emp, beh, vis correlations overlaid, one panel per subject.

**Existing infrastructure to reuse:**
- Inference loop: `compare_models_fixrsvp.py` (lines ~245–264) or the cleaner version in `subspace_residual.py:_run_inference`.
- Affine rescale: `eval.eval_stack_utils.rescale_rhat`.
- PSD projection: `VisionCore.covariance.project_to_psd`.
- Subspace overlap: `VisionCore.subspace.symmetric_subspace_overlap`.

**Estimated cost:** A few hours including figure work, no model retraining. The first hour is extending `subspace_residual.py` (or writing `image_computable_fem.py`) to cache rhat tensors per session and compute the per-time covariances. The model inference for all 24 sessions has already been done once for the pilot — if we add tensor caching, future runs are I/O-bound.

---

#### A2. Eye-conditional PSTH sharpening (image-computable)

**Question addressed:** The empirical Fig 1g/h shows that re-ordering trials by eye-trajectory similarity sharpens single-cell PSTHs substantially. Does the behavior model reproduce this sharpening? Does vision-only fail to?

**Inputs:**
- Same as A1: rhat_beh, rhat_vis, robs per session on fixRSVP.

**Algorithm:**

```
For each cell c:
    For each method M in {empirical, beh, vis}:
        For each stim time t:
            For each pair of trials (i, j):
                d_ij = RMS distance between eye trajectories at lead-in window
            Group trial pairs into bins by d_ij (same binning as Fig 1h)
            Compute conditional mean response E[r_c(t) | bin]
        Compute "sharpness ratio":
            σ²(within bin) / σ²(across bins)
            or equivalently, fraction of variance explained by binning
    Compare sharpness ratio across methods.
```

A simpler operationalization: for each cell and each model output, compute the variance of the trial-average response across bins (signal-like) vs the variance within bins (residual). The ratio is the eye-conditional sharpening that model produces.

**Metric:** ΔSharpness = sharpness(model) − sharpness(unconditioned). Empirical, beh, vis.

**Expected outcome:**
- Empirical shows substantial sharpening (already documented in Fig 1).
- Behavior model produces a similar magnitude of sharpening — by construction it can, and the question is whether it does.
- Vision-only produces partial sharpening from shifter-residual variance.

**Decision tree:**
| Outcome | Interpretation |
|---|---|
| beh sharpening ≈ empirical sharpening; vis ≪ both | Behavior model is the image-computable analog of the Fig 1g/h finding |
| beh ≈ vis | Behavior model not adding the sharpening → again contradicts ΔBPS |
| beh > empirical | Model is overfitting eye state → unlikely given the regularization but worth checking |

**Why this complements A1:** A1 is at the population covariance level. A2 is at the single-cell response-shape level. If A1 lands and A2 also lands, the manuscript can show both: "behavior model recovers shared variance" and "behavior model recovers single-cell eye-conditioning."

**Existing infrastructure:**
- Fig 1 sharpening: status partial (`ryan/fig1`).
- Eye-trajectory binning code: see `compare_models_fixrsvp.py` for trial gathering on fixRSVP; binning logic likely needs to be adapted from Fig 2's "match eye trajectories" code (`VisionCore.covariance.bin_pairs_by_distance`, `compute_eye_distances`).

**Estimated cost:** A few hours; reuses much of the pair-distance machinery from `VisionCore.covariance`.

---

#### A3. Information geometry of the behavior modulation

**Question addressed:** Is the across-trial modulation predicted by the behavior model **information-limiting** (parallel to the population stimulus-tuning gradient) or **information-preserving** (orthogonal to it)? This is the bridge to Fig 5.

**Inputs:**
- $\hat\Sigma_{\mathrm{FEM}}^{\text{beh}}$ from A1.
- Population stimulus-tuning gradient: PSTH eigenvectors as a proxy, or computed directly from $\partial \hat r / \partial s$ via the model.

**Algorithm (simpler version):**

```
For each session:
    Get eigendecomposition of Σ_PSTH (from fig2 cache):  Σ_PSTH = V_P Λ_P V_P^T
    Get eigendecomposition of Σ_pred^beh (from A1):       Σ_pred^beh = V_B Λ_B V_B^T

    For each k in {1, ..., 10}:
        f_parallel(k) = trace(V_P[:, :k]^T Σ_pred^beh V_P[:, :k]) / trace(Σ_pred^beh)
        f_orthogonal(k) = 1 - f_parallel(k)

    Decoding test:
        Train a linear stimulus-time decoder on rhat_beh, rhat_vis, robs.
        Compare cross-validated accuracy.
```

**Key plot:** f_parallel(k) as a function of k for behavior-model predicted modulation. Compare to:
- Empirical Σ_FEM aligned to Σ_PSTH (already in Fig 2k).
- A random-direction null.

**Expected outcomes & interpretations:**
- f_parallel large (~0.7) → behavior modulation lives in stimulus-tuned dimensions → consistent with Fig 2 finding that Σ_FEM is PSTH-aligned. Information-limiting in the strict Moreno-Bote sense.
- f_parallel small → behavior modulation orthogonal to stimulus tuning → information-preserving. Would suggest the modulation gates / tags responses without corrupting decoding.

**Stronger version — direct decoding:** For each session, decode held-out stim identity (use psth_inds as the class label, restricted to N most-frequent classes) from each model's predicted activity and from real activity. Comparison decoder: one trained on robs (upper bound), one on rhat_beh, one on rhat_vis.

**Decision tree:**
| Outcome | Interpretation | Manuscript implication |
|---|---|---|
| Decode(rhat_beh) ≈ Decode(robs) >> Decode(rhat_vis) | Behavior model recovers stimulus information that vision-only loses | Strong Fig 4 → Fig 5 link |
| Decode(rhat_beh) ≈ Decode(rhat_vis) | Behavior modulation doesn't change stimulus decodability | Fig 4 stands alone, Fig 5 needs other support |
| Decode(rhat_beh) < Decode(rhat_vis) | Behavior modulation hurts stimulus decoding | Surprising; would indicate info-limiting modulation, possibly motor/arousal |

**Estimated cost:** Half-day for the geometric decomposition; full day if we add the gradient-based stimulus-tuning direction (requires running gradients on the model w.r.t. its stimulus input).

---

### Tier 2 — Mechanism

#### B1. Channel ablations

**Question addressed:** Which behavior subchannel — eye_pos or eye_vel — drives the FEM recovery? This pins the mechanism (gain field vs motor signal).

**Algorithm:**
At inference time, zero out the relevant slice of the behavior tensor before running through the behavior model.

The behavior tensor is 22-d, ordered as `[eye_vel × 20 channels, eye_pos × 2 channels]` (per `multi_basic_120.yaml:25-50`; verify ordering in `models/data/loading.py:559-564`).

Three ablation conditions:
1. `behavior[..., :20] = 0` → eye_vel zeroed, eye_pos preserved.
2. `behavior[..., 20:] = 0` → eye_pos zeroed, eye_vel preserved.
3. `behavior[...] = 0` → both zeroed (sanity check; should approximate vision-only).

Per ablation, recompute:
- Per-cell ΔBPS (vs vision-only baseline).
- $\hat\Sigma_{\mathrm{FEM}}^{\text{beh-ablated}}$ alignment with empirical Σ_FEM (A1 metrics).

**Decision tree:**
| Outcome | Interpretation |
|---|---|
| Eye_pos ablation kills ΔBPS | eye_pos drives the channel → gain-field story (Trotter, Galletti) |
| Eye_vel ablation kills ΔBPS | eye_vel drives it → motor / peri-saccadic |
| Both retain ~50% | Mixed; both contribute |

**Existing infrastructure:** `run_model` from `eval.eval_stack_utils`. Need a small wrapper that injects an ablation mask into `batch['behavior']` before calling the model.

**Estimated cost:** Half-day. Basically a re-run of the inference loop with three additional behavior tensor variants, plus re-running A1 metrics on each.

---

#### B2. Stimulus-only / behavior-only baselines

**Question addressed:** Does behavior carry V1-relevant information *independent of the stimulus*? Is the behavior contribution multiplicative (gain on stim-driven activity) or additive (stim-independent)?

**Algorithm:**

Two ablation conditions at inference:
1. **Stimulus zeroed in behavior model**: pass the mean luminance / blank stimulus, real behavior. Output = behavior-only contribution + bias.
2. **Behavior zeroed in behavior model** (= condition 3 of B1): pass real stim, zero behavior.

Compare the resulting BPS to vision-only's BPS.

A cleaner version requires **training a behavior-only model** (no stim input). That's a heavier ask. The inference-time ablation is cheap and produces the upper bound.

**Decision tree:**
| Outcome | Interpretation |
|---|---|
| Stim-zero ablation BPS > 0 | Behavior carries stim-independent V1 info → true extraretinal |
| Stim-zero ablation BPS ≈ 0 | Behavior modulates only stim-driven activity → gain-like |

**Caveat:** The behavior model wasn't trained for the stim-zeroed condition. The result is suggestive but should be confirmed with a properly trained behavior-only model before going in the manuscript.

---

#### B3. Per-cell dependency mapping

**Question addressed:** What is the *shape* of each cell's behavior dependency? Are there interpretable clusters?

**Algorithm:**

For each cell c:
- Fix a representative stimulus s₀ (e.g., the trial-average input at a fixed stim time).
- Vary eye_pos over a 2D grid (e.g., 11×11 grid centered on the fixation window).
- Compute rhat_beh(s₀, eye_pos = grid_point) for each grid point.
- Visualize as a 2D heatmap → eye-position gain field for cell c.

Repeat for eye_vel:
- Fix s₀, eye_pos = mean.
- Vary eye_vel along the basis dimensions (e.g., set one basis function active at a time, sweep its amplitude).
- Plot rhat_beh as a function of eye_vel basis amplitude.

**Outputs:**
- Eye-position gain map per cell. Cluster cells by similarity (e.g., flat → no gain field, monotone → linear gain, smooth bump → tuned, sharp → unstable).
- Eye-velocity dependence per cell.

**Decision tree:**
| Cluster pattern | Interpretation |
|---|---|
| Smooth monotone gain in eye_pos | Classical gain field |
| Bump-shaped or RF-shifted gain | RF translation (eye-position-dependent RF location) |
| Sharp eye_vel dependence at large velocities | Peri-saccadic modulation |
| No dependence | Cell is not behavior-modulated; ΔBPS contribution is 0 |

**Estimated cost:** Day or two. Lots of plotting; the actual inference is cheap because we only sweep behavior at fixed stim.

---

### Tier 3 — Temporal localization

#### C1. Time-resolved ΔBPS

**Question addressed:** *When* during a fixation does the behavior model help most? Different temporal signatures imply different mechanisms.

**Algorithm:**

Bin time-from-fixation-onset (e.g., 0–100 ms, 100–200 ms, 200–500 ms, 500+ ms). In each bin, recompute BPS for both models, and plot ΔBPS(time-from-fix-onset).

Repeat for time-from-saccade-onset (negative = pre-saccadic, positive = post-saccadic).

**Expected signatures:**
| Signature | Mechanism |
|---|---|
| ΔBPS spikes at fixation onset, decays | Initial retinal-slip / shifter residual at fixation onset |
| ΔBPS uniform across fixation | Steady-state gain modulation |
| ΔBPS spikes peri-saccadic | Motor / peri-saccadic modulation |
| ΔBPS oscillates with drift cycle | Drift-related modulation (rare but possible) |

**Existing infrastructure:** `compare_models.py` already has perisaccadic figures; this is an extension that adds the time-resolved ΔBPS quantification.

---

#### C2. Saccade-triggered residual

**Question addressed:** What does the behavior model capture in the peri-saccadic interval that vision-only misses?

**Algorithm:** Already partially in `compare_models.py`. Compute, per cell, peri-saccadic mean responses for:
- empirical robs
- rhat_beh
- rhat_vis

Plot all three on the same axes per cell. Then plot the **gap** (empirical − rhat_vis) and (rhat_beh − rhat_vis) — these should match if the behavior model captures what vision misses in the peri-saccadic window.

**Estimated cost:** Mostly already done; needs an additional comparison panel.

---

### Tier 4 — Controls

#### D1. Shifter-residual control

**Question addressed:** How much of the behavior-model advantage is "the shifter is imperfect" vs "true extraretinal modulation"?

**Algorithm:**

This is harder. One approach:
1. Estimate the shifter's per-session residual error magnitude. One proxy: the variance of single-cell STAs across trial-blocks at fixed nominal eye position. Larger spread → larger shifter error.
2. Simulate a vision-only model with imperfect input: jitter the shifted stim by a Gaussian noise calibrated to the residual magnitude.
3. Compute $\hat\Sigma_{\mathrm{FEM}}^{\text{vis-imperfect}}$.
4. Compare its alignment with empirical Σ_FEM to behavior model's recovery.

If the simulated imperfect-vision model recovers most of empirical Σ_FEM → behavior advantage is shifter-residual cleanup. If not → real extraretinal contribution.

**Estimated cost:** Day or two. The shifter-residual estimation is non-trivial.

---

#### D2. Random-subspace null

**Question addressed:** Is the observed Σ_pred^beh alignment with Σ_PSTH / Σ_FEM far above what random subspaces would give?

**Algorithm:** Sample U_random uniformly on the Stiefel manifold. Compute capture(Σ_pred^beh in U_random) for k = 1, ..., 10. Repeat 1000 times for a null distribution.

**Why bother:** Replaces the broken trial-shuffle null in `subspace_residual.py`. Cheap (analytic), and gives the proper structural baseline. Even if A1 lands cleanly, having the right null in the supplementary materials adds rigor.

**Estimated cost:** A couple hours.

---

## 5. Order of operations & decision tree

```
                             ┌─────────────────────────────┐
                             │  A1: Σ_FEM recovery test    │
                             │  (gating analysis)          │
                             └──────────────┬──────────────┘
                                            │
                  ┌─────────────────────────┴─────────────────────────┐
                  │                                                     │
        beh ≈ empirical >> vis                            beh ≈ vis  OR  no recovery
                  │                                                     │
                  ▼                                                     ▼
   Tier 1 continues:                                  STOP — debug pipeline.
   A2 (single-cell sharpening recovery)               Possible: rescaling broken,
   A3 (information geometry / decoding)               cache misalignment, or
                                                      finding doesn't actually
                  ▼                                   reflect a structured residual.
   Tier 2 (mechanism):
   B1 (channel ablations)
   B2 (stim/behavior-only baselines)
   B3 (per-cell dependency maps)

                  ▼
   Tier 3 (temporal):
   C1 (time-resolved ΔBPS)
   C2 (saccade-triggered residual)

                  ▼
   Tier 4 (controls):
   D1 (shifter-residual control)   — go in supplementary
   D2 (random-subspace null)
```

**Critical path to Fig 4:** A1 → B1 → A2/A3.

If A1 lands, A1 alone is a strong main panel. B1 gives the mechanism subpanel ("eye_pos drives this" or "eye_vel drives this"). A2/A3 give the link to Fig 5 (function).

If A1 doesn't land, return to the Layer 1 question: what is the behavior input actually delivering? B1 and B2 become diagnostic rather than mechanistic.

---

## 6. Files, caches, infrastructure

### 6.1 Existing scripts in `VisionCore/ryan/behavior-vs-vision/`

- `compare_models.py` — cross-stim-class BPS histograms + perisaccadic comparisons. Original discovery script.
- `compare_models_fixrsvp.py` — fixRSVP-specific, with affine rescaling + ccnorm + r². Per-cell metrics cached at `outputs/cache/behavior_vs_vision_fixrsvp.pkl` (bps, ccnorm, ccabs, ccmax, ve, no rhat tensors).
- `subspace_residual.py` — Σ_Δ subspace alignment pilot. Cache at `outputs/cache/behavior_vs_vision_residual_subspace.pkl`. **Has the inference loop we'll reuse**: `_run_inference` is clean and supports an arbitrary dict of models.
- `figures/` — saved per-session and summary figures from the pilot.
- This file: `STRATEGY.md`.

### 6.2 Caches likely to be useful

- `outputs/cache/fig2_decomposition.pkl` — per-session: `mats[w_idx]` containing PSTH, FEM, Intercept (= rate cov), Total covariance matrices; `neuron_mask`; `meta`. 4 windows: bins [1, 2, 4, 8] ≈ ms [8.3, 16.7, 33.3, 66.7].
- `outputs/cache/fig3_digitaltwin.pkl` — Fig 3 results for the published digital twin model on free-viewing.
- `outputs/cache/behavior_vs_vision_fixrsvp.pkl` — fixRSVP per-cell metrics (no rhat).
- `outputs/cache/behavior_vs_vision_residual_subspace.pkl` — Σ_Δ pilot results (no rhat).

**For A1, we need to cache the rhat tensors themselves.** Either extend `compare_models_fixrsvp.py` to dump per-session `(robs, rhat_beh, rhat_vis, dfs, psth_inds)` tensors, or add the same to `subspace_residual.py`. Storage estimate: 24 sessions × ~80 trials × 120 bins × ~100 cells × 4 bytes × 2 models ≈ 200 MB. Manageable.

### 6.3 Library APIs

- `VisionCore.covariance` — `project_to_psd`, `run_covariance_decomposition`, `bagged_split_half_psth_covariance`, `estimate_rate_covariance`, `compute_eye_distances`, `bin_pairs_by_distance`, `align_fixrsvp_trials`, `cov_to_corr`, `get_upper_triangle`.
- `VisionCore.subspace` — `participation_ratio`, `symmetric_subspace_overlap`, `directional_variance_capture`, `project_to_psd`.
- `VisionCore.stats` — `bootstrap_mean_ci`, `bootstrap_paired_diff_ci`, `fisher_z`, `wilcoxon_signed_rank`, `fdr_correct`.
- `eval.eval_stack_utils` — `load_single_dataset`, `run_model`, `rescale_rhat`, `ccnorm_split_half_variable_trials`, `bits_per_spike`.
- `eval.eval_stack_multidataset` — `load_model`, `evaluate_model_multidataset`.

### 6.4 Model checkpoints

Hard-coded in `compare_models.py`, `compare_models_fixrsvp.py`, `subspace_residual.py`:

- Behavior: `/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120/2026-03-31_11-33-32_learned_resnet_concat_convgru_gaussian/learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4`
- Vision-only: `/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120/2026-04-13_11-10-15_learned_resnet_none_convgru_gaussian/learned_resnet_none_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4`

The `find_best_ckpt` helper (in all three scripts) selects the highest val_bps_overall checkpoint.

### 6.5 Data conventions

- fixRSVP trial gathering: in each script, `train_data.get_dataset_inds('fixrsvp')` is called per session via `load_single_dataset`. The relevant covariates are `trial_inds` (which trial each sample belongs to) and `psth_inds` (which "stim time" within the repeated sequence).
- Fixation mask: `np.hypot(eyepos[:,0], eyepos[:,1]) < 1.0` (1° fixation window).
- Min fixation duration: 20 bins (≈ 167 ms at 120 Hz).
- Min total spikes per cell to include: 200 (matches `compare_models_fixrsvp.py` and `subspace_residual.py`).
- Counting window for the fig2 analog: use `FIG2_WINDOW_IDX = 0` (≈ 8.33 ms, the smallest window, matching the model's 1/120 s timestep).

### 6.6 Where to put new scripts

- `VisionCore/ryan/behavior-vs-vision/image_computable_fem.py` — A1 implementation.
- `VisionCore/ryan/behavior-vs-vision/conditional_psth_sharpening.py` — A2.
- `VisionCore/ryan/behavior-vs-vision/info_geometry.py` — A3.
- `VisionCore/ryan/behavior-vs-vision/channel_ablations.py` — B1, B2.
- `VisionCore/ryan/behavior-vs-vision/per_cell_dependency.py` — B3.
- `VisionCore/ryan/behavior-vs-vision/temporal_dynamics.py` — C1, C2.
- `VisionCore/ryan/behavior-vs-vision/controls.py` — D1, D2.

Figures land in `outputs/figures/behavior-vs-vision/`. Caches in `outputs/cache/behavior_vs_vision_*.pkl`.

### 6.7 Anchor existing manuscript Methods text

When writing prose for the manuscript, anchor the new analyses to the existing Methods sections in `main.tex`:

- **Image-computable model** is described at `Methods → Digital Twin modeling`.
- **Σ_FEM definition** is at `Methods → Decomposition of Neural Variability via the Law of Total Covariance` (the appendix).
- **Σ_FEM estimator** is at `Methods → Robust Estimation of Eye-Movement-Dependent Covariance`.
- **Subspace alignment** is at `Methods → Alignment between stimulus-locked and eye-movement covariance`.

The image-computable Σ_FEM analog (A1) should be presented as a direct model-based version of these, reusing the same notation.

---

## 7. Open methodological points (for next session)

1. **Stim-time alignment across trials in fixRSVP.** Need to confirm that `psth_inds` gives the same world-frame stim element across repeats. Spot-check on one session before scaling A1.
2. **Affine rescale interaction with covariance.** A1 needs rhat in spike-count units. Affine rescale per cell adjusts magnitude and offset. Per-cell offsets vanish under cov; per-cell gains scale the diagonal but also scale off-diagonal pairwise covs. Verify the rescaling does what we want — possibly safer to use unrescaled rhat for the *direction* analysis and rescaled rhat for the *magnitude* analysis.
3. **Sample-size bias on per-time covariances.** With ~50–100 trials at fixed stim time and up to ~150 cells, the per-time cov is rank-deficient. Pool across many stim times to get adequate sampling. May need to pool by lag-bin if shorter time windows are rank-deficient.
4. **fixRSVP vs frozen-image stimulus.** The empirical Σ_FEM in the fig2 cache was estimated on the frozen-image stimulus, NOT fixRSVP. For a strict apples-to-apples comparison, the fig2 decomposition should be re-run on fixRSVP. This is mostly a matter of pointing the existing fig2 pipeline at fixRSVP samples; the estimator is unchanged. Decide early whether to use the existing fig2 cache (treating Σ_FEM as a session-level subspace property) or re-estimate per fixRSVP. The pilot used the existing cache; A1 should ideally re-estimate.
5. **Cell intersection between fig2 and fixRSVP.** Fig 2 filters by spike count + finite cov on the frozen-image stim; fixRSVP filters by spike count on fixRSVP. Use the intersection (`subspace_residual.py:_intersect_neuron_masks`). This already loses a few cells per session.

---

## 8. What would convince a skeptical reviewer

For the manuscript, the strongest version of the Fig 4 result would include:

- **A1**: cross-session magnitude / eigenvector / pairwise-correlation comparison showing $\hat\Sigma_{\mathrm{FEM}}^{\text{beh}} \approx \Sigma_{\mathrm{FEM}}^{\text{empirical}}$ and $\hat\Sigma_{\mathrm{FEM}}^{\text{vis}}$ substantially smaller. This is the headline panel.
- **B1**: ablation showing the recovery is driven primarily by one or both of eye_pos / eye_vel, with a clean ΔBPS attribution.
- **D1**: shifter-residual control showing that simulated imperfect vision-only does NOT reproduce the recovery — i.e., the behavior advantage is not explained by shifter cleanup alone.
- **A3**: information-geometry panel showing that the recovered modulation is along stimulus-tuned directions (consistent with Fig 2) but does not collapse stimulus decoding (the decoding test).

If those four pieces line up, Fig 4 is strong, and the chain Fig 2 → Fig 4 → Fig 5 closes.

---

## 9. Living notes

This file is the strategy at the time of writing. Findings and pivots should be appended to `fem-v1-fovea/lab-journal.md` (existing). Updates that materially change the strategy should be reflected here too.
