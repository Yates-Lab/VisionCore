---
title: "A fraction of what? Deriving $r^2/R^2_{\\max}$ end to end"
subtitle: "What figure 3E-extended's row B actually computes, where it is exact, and what it inherits — with the numbers"
author: "fem-v1-fovea methods note (figure3e_extended)"
date: "2026-07-28"
header-includes: |
  <style>
  .numbered-equation { position: relative; }
  .numbered-equation .eqno {
    position: absolute; right: 0.5em; top: 50%;
    transform: translateY(-50%);
  }
  table { border-collapse: collapse; margin: 1.2em 0; font-size: 0.85em; }
  th, td { border-bottom: 1px solid #ddd; padding: 0.25em 0.7em; text-align: left; }
  th { border-bottom: 2px solid #999; }
  code { font-size: 0.88em; }
  blockquote { border-left: 3px solid #ccc; padding-left: 1em; color: #444; }
  </style>
---

# Summary

Row B of `figure3e_extended` reports, per unit,

$$
\frac{r^2_{\text{cond}}}{R^2_{\max}},
\qquad
r^2 = 1 - \frac{\operatorname{Var}[\hat y - y]}{\operatorname{Var}[y]},
\qquad
R^2_{\max} = \frac{\operatorname{diag} C_{\text{rate}}}{\operatorname{diag} C_{\text{total}}} ,
\tag{1}
$$

where the numerator is the single-trial $r^2$ figure 3 panel D already plots and
the denominator is figure 2's per-unit rate-variance fraction, read at the
model's own 120 Hz bin. This note derives both halves from scratch, states the
exact conditions under which the ratio is a *fraction of explainable variance*,
and reports what I measured today when I checked each condition against the
production caches.

**The short version.** The bound behind $R^2_{\max}$ is exact and assumes far
less than one might fear — only that the spike-count residual is conditionally
mean-zero given the conditioning variables, not that it is Poisson. Every
premise the bound needs is satisfied here, and two of them I verified directly
rather than assumed: the fixRSVP image sequence really is frozen across trials
(so "time in trial" *is* the screen stimulus), and fixRSVP is genuinely held out
of the twin's training set. Three of the listed caveats in the folder README
turned out to be numerically negligible when measured; one — the heavy right
tail created by small denominators — is real and is the only thing I would
change before putting this on figure 3.

**The numbers, population = 972 units** (19 sessions, fig. 2 inclusion, finite
ceiling):

| condition | median $r^2$ | median $r^2/R^2_{\max}$ | frac. $>1$ | frac. $<0$ |
|---|---|---|---|---|
| trial average (PSTH) | 0.0216 | 0.190 | 0.016 | 0.024 |
| full | 0.0396 | **0.350** | 0.061 | 0.051 |
| ablated | 0.0362 | 0.326 | 0.050 | 0.058 |
| stabilized within window | 0.0361 | 0.325 | 0.051 | 0.048 |
| stabilized within trial | 0.0163 | 0.153 | 0.013 | 0.157 |
| stabilized across trials | 0.0029 | 0.024 | 0.002 | 0.421 |
| stabilized + ablated | $-0.0025$ | $-0.021$ | 0.002 | 0.569 |

Median $R^2_{\max} = 0.1235$ in this population (IQR $[0.074,\,0.195]$; 5th–95th
$[0.025,\,0.341]$).

---

# 1. Setting and notation

I follow the notation of `../methods_eyepos_matching/writeup.md` so the two
notes compose. Let $Y^i_c(t)$ be the spike count of unit $c$ on trial $i$ in
analysis time bin $t$ (time *within* the fixRSVP trial, i.e. the PSTH index),
let $e$ denote gaze, and write the conditional rate

$$
\lambda_c(t, e) \;=\; \mathbb{E}\!\left[\, Y_c \,\middle|\, t,\, e \,\right].
\tag{2}
$$

Everything below is per unit, so I drop $c$ where it is not needed.

## 1.1 What one "sample" is

Both halves of (1) are computed on single 120 Hz bins of the same fixRSVP data,
but through two independent pipelines that make slightly different selections.
Laying them side by side is the whole ball game, because the ratio is only a
fraction of something if numerator and denominator describe the same
distribution:

| | numerator ($r^2$, `_ext_data.py`) | denominator ($R^2_{\max}$, `covariance_decomposition/`) |
|---|---|---|
| source | model's own datasets, via the checkpoint | `multi_basic_120_long.yaml` |
| fixation mask | $\lVert e\rVert < 1.0^\circ$ (`_ext_data.py:174`) | $\lVert e\rVert < 0.5^\circ$ (`data_loading.py:34`) |
| trial filter | $\ge 20$ fixation bins | $\ge 20$ fixation bins |
| time bins kept | first 120 per trial | first 120 per trial |
| bin selection | every bin with `dfs > 0` | strided windows inside valid runs of $\ge 36$ bins |
| unit filter | $>200$ spikes, then fig. 2 inclusion | $>0$ spikes, then fig. 2 inclusion |
| counting window | 1 bin (8.33 ms) | 1 bin, `PRODUCTION_WINDOW_BINS = 1` |

The join is on `(session, neuron_id)` where `neuron_id` is the dataset's own
column index in both pipelines (`tests/test_ceiling.py:52` pins this), so the
key is well defined as long as both prep paths see the same sorted units in the
same order. The two selections differ in the fixation radius and in the bin
striding; §5.2 measures how much that matters (answer: 2–4 %, in the direction
that makes $R^2_{\max}$ very slightly conservative).

## 1.2 What the twin conditions on

This matters for the bound, so it is worth being exact rather than gesturing at
"stimulus and gaze".

*The screen stimulus is a function of time in trial.* I checked this directly
rather than inferring it from PSTH reliability. On `Allen_2022-02-16` (77
trials), reconstructing each trial's flash history through
`FixRsvpTrial.image_ids` / `.positions` and aligning by within-trial sample
index gives **exactly one distinct sequence**: identical image ids and identical
draw positions at every index, across all trials long enough to compare. The
session collapses to 32 distinct screen contents. So conditioning on $t$ is
conditioning on the image on the screen, and the close-pair estimator's pairing
of two trials at the same $t$ pairs two presentations of the *same* stimulus.
Had the RSVP sequence been re-randomized per trial, $C_{\text{rate}}$ would
have measured the rate variance marginalized over image identity and the whole
ceiling argument would collapse. It does not.

*The behavior input is a function of gaze.* The `behavior` tensor (42 channels)
is built entirely from `eyepos`: a 10-cosine **acausal** temporal-basis
embedding of eye velocity (`history_bins: 50`, `causal: false`, peaks
30–200 ms) passed through a split-ReLU, concatenated with the raw 2-D eye
position. No pupil, no reward, no non-gaze state. So the twin's input is
$(\text{retinal stimulus history},\ f(\text{gaze trajectory}))$, and the retinal
stimulus history is itself the screen stimulus cropped at gaze. The twin is
therefore a *measurable function of $(t, e_{\cdot})$*, which is precisely what
the bound in §2 requires.

*The gaze window is wide.* The retinal input spans 33 frames (275 ms of causal
history) and the behavior basis reaches roughly $\pm 200$ ms, i.e. into the
future. The ceiling's close-pair matching, at `window_bins = 1`, matches
trajectories only over $t_{\text{hist}} + t_{\text{count}} = 2$ bins (16.7 ms).
The twin conditions on strictly more than the ceiling does. §2.3 turns this into
the single most important caveat: $R^2_{\max}$ as computed is a *lower bound* on
the twin's own ceiling.

*fixRSVP is held out.* `multi_basic_120_long.yaml` trains on
`types: [backimage, gaborium, gratings]`; `fixrsvp` is appended only at
evaluation time (`eval/eval_stack_utils.py:296`). The $r^2$ is therefore an
honest generalization number, not a training fit. This is worth stating in the
caption if row B becomes panel D, because a reader who sees "fraction of
explainable variance captured" will immediately ask.

---

# 2. The denominator: deriving the ceiling

## 2.1 The bound

Let $z$ collect the conditioning variables — here $z = (t, e_\cdot)$, time in
trial and the gaze trajectory — and let $\lambda = \mathbb{E}[y \mid z]$. The
law of total variance splits the observed count variance into rate variance and
private noise:

$$
\operatorname{Var}(y) \;=\; \operatorname{Var}(\lambda) \;+\; \mathbb{E}\big[\operatorname{Var}(y \mid z)\big].
\tag{3}
$$

Now take *any* predictor $\hat y = g(z)$ that is a function of the conditioning
variables. Decompose the error as $y - \hat y = (y - \lambda) + (\lambda - \hat y)$
and expand:

$$
\mathbb{E}\big[(y-\hat y)^2\big]
= \mathbb{E}\big[(y-\lambda)^2\big]
+ 2\,\mathbb{E}\big[(y-\lambda)(\lambda-\hat y)\big]
+ \mathbb{E}\big[(\lambda-\hat y)^2\big].
\tag{4}
$$

The cross-term vanishes by the tower rule, because $\lambda$ and $\hat y$ are
both $z$-measurable:

$$
\mathbb{E}\big[(y-\lambda)(\lambda-\hat y)\big]
= \mathbb{E}\Big[\ \underbrace{\mathbb{E}[\,y-\lambda \mid z\,]}_{=\,0}\ (\lambda-\hat y)\Big] = 0 .
\tag{5}
$$

This is the only probabilistic assumption in the entire derivation: the residual
is conditionally mean-zero given $z$. It is true by the *definition* of
$\lambda$. Nothing here assumes Poisson, or independence across bins, or
stationarity. Substituting (5) into (4),

$$
\mathbb{E}\big[(y-\hat y)^2\big]
= \mathbb{E}\big[(\lambda-\hat y)^2\big] + \mathbb{E}\big[\operatorname{Var}(y\mid z)\big]
\;\ge\; \mathbb{E}\big[\operatorname{Var}(y\mid z)\big],
\tag{6}
$$

with equality **iff** $\hat y = \lambda$ almost surely. Dividing by
$\operatorname{Var}(y)$ and using (3),

$$
r^2 \;=\; 1 - \frac{\mathbb{E}[(y-\hat y)^2]}{\operatorname{Var}(y)}
\;\le\; 1 - \frac{\mathbb{E}[\operatorname{Var}(y\mid z)]}{\operatorname{Var}(y)}
\;=\; \frac{\operatorname{Var}(\lambda)}{\operatorname{Var}(y)}
\;=:\; R^2_{\max}.
\tag{7}
$$

That is the whole thing. $R^2_{\max}$ is the $r^2$ of the *oracle* predictor
that knows $\lambda$ exactly, and no function of $z$ can beat it.

## 2.2 Two ways to read $R^2_{\max}$

Because figure 2 also estimates $C_{\text{psth}}$, the ceiling factors:

$$
R^2_{\max}
= \underbrace{\frac{\operatorname{diag}C_{\text{psth}}}{\operatorname{diag}C_{\text{total}}}}_{\text{ideal-PSTH share}}
+ \underbrace{\frac{\operatorname{diag}(C_{\text{rate}} - C_{\text{psth}})}{\operatorname{diag}C_{\text{total}}}}_{\text{FEM share}}
= R^2_{\max}\big[\alpha + (1-\alpha)\big].
\tag{8}
$$

Empirically, median $\alpha = 0.362$ in this population (so median
$1-\alpha = 0.638$, figure 2's headline). A *noise-free* trial-average predictor
would therefore land at $r^2/R^2_{\max} = \alpha \approx 0.36$; the full twin's
median is $0.350$, and **48.5 %** of units have $r^2_{\text{full}}$ above their
own $\alpha R^2_{\max}$, i.e. above what any purely stimulus-locked predictor
could ever achieve. (Against the *measured* leave-one-out PSTH, which is noisy,
the twin wins on 76.1 % of units.) This is a good way to state the panel-D
result once the axis is normalized, and it is not available on the raw $r^2$
axis.

## 2.3 What the bound does **not** say

Two mismatches between the $z$ of the estimator and the $z$ of the model, in
opposite directions:

**(a) The estimator conditions on less than the model, so the measured ceiling
is a lower bound.** $C_{\text{rate}}$ is estimated by matching gaze
*trajectories* over 2 bins to within 0.05° RMS. Write $z_{\text{est}}$ for that
coarser conditioning set and $z_{\text{mod}} \supseteq z_{\text{est}}$ for the
model's. Since $\operatorname{Var}(\mathbb{E}[y\mid z_{\text{est}}]) \le
\operatorname{Var}(\mathbb{E}[y \mid z_{\text{mod}}])$ by the tower rule, the
ceiling the model actually faces is at least the measured one:

$$
R^2_{\max}\big[z_{\text{est}}\big] \;\le\; R^2_{\max}\big[z_{\text{mod}}\big].
\tag{9}
$$

So $r^2/R^2_{\max} > 1$ is *legal*, not a bug, and the line at 1 must not be
drawn as a hard ceiling. The finite 0.05° tolerance pushes the same way (paired
samples do not share a rate exactly), as does any dependence of $\lambda$ on
gaze history beyond 16.7 ms — which, for a V1 model with a 275 ms integration
window, is not a small thing. 6.1 % of units exceed 1 in the full condition.

**(b) Shared slow drift pushes the other way.** The close-pair product
$Y^i Y^j$ is unbiased for $\lambda^2$ only if the two trials' residuals are
independent. They are distinct trials, so simultaneous noise cannot contaminate
them — but slowly varying excitability (arousal, adaptation, drift in isolation
quality) shared between nearby trials survives the pairing and inflates
$C_{\text{rate}}$, hence inflates $R^2_{\max}$ and deflates the reported
fraction. Pairs are enumerated within a time-in-trial group with no restriction
on trial separation, so nearby-in-session trials contribute. This is the
standard McFarland-family concern and it is not corrected anywhere in the
pipeline. I have not quantified it; a trial-lag-restricted re-estimate of
$C_{\text{rate}}$ would.

---

# 3. Estimating the two diagonals

Neither term in $R^2_{\max}$ is observable, since we never see $\lambda$. Both
come out of the figure 2 stage-1 decomposition; `ceiling.py` does no estimation,
it only reshapes.

## 3.1 Windowing

`extract_windows` (`VisionCore/covariance.py:90`) strides samples inside
contiguous valid runs of $\ge 36$ bins. For a counting window of $t_{\text{count}}$
bins after a history of $t_{\text{hist}} = \max(t_{\text{hist,bins}}, t_{\text{count}})$
bins, each sample carries: a count vector (summed over the count window), a gaze
trajectory over all $t_{\text{hist}} + t_{\text{count}}$ bins, and
$T_{\text{idx}}$ = the count-window start bin. Stride is $t_{\text{count}}$.

At the production setting `window_bins = 1`, with `t_hist_ms = 10` giving
$t_{\text{hist,bins}} = \lfloor 10 / 8.33 \rfloor = 1$: the count is a **single
120 Hz bin**, the trajectory is **2 bins**, and the stride is 1 bin — so every
valid bin in a long-enough run becomes a sample. This is the same bin the $r^2$
is computed on, which is the point.

## 3.2 $\operatorname{diag} C_{\text{total}}$

The legacy unweighted sample covariance of the windowed counts,
`np.cov(X.T, ddof=1)` over samples with a finite row sum
(`decompose.py:62`). No importance weights, no time-bin weights. Note the
asymmetry with the numerator of $R^2_{\max}$, which *is* weighted — §6.6.

## 3.3 $\operatorname{diag} C_{\text{rate}}$: close pairs

The engine is that observation noise is independent across distinct trials, so
for $i \neq j$ at the same $t$,

$$
\mathbb{E}\big[Y^i(t)\,Y^j(t)\,\big|\, e_i, e_j\big] = \lambda(t, e_i)\,\lambda(t, e_j).
\tag{10}
$$

Restricting to pairs whose gaze trajectories nearly coincide drives
$e_i \approx e_j \approx e$, so the product estimates $\lambda(t,e)^2$ with the
private noise already removed. "Nearly coincide" is RMS trajectory distance
(`_rms_traj_close_pairs`, `covariance.py:604`): the L2 distance of the flattened
$(T\times 2)$ trajectory vectors, scaled by $1/\sqrt{T}$ so the threshold keeps
per-bin distance units, thresholded at $\varepsilon = 0.05^\circ$.

**The sampling-distribution problem.** Close pairs are not drawn from the
viewing distribution $p(e)$ — they are drawn from something close to $p(e)^2$,
which over-represents the centre of the fixation cluster where pairs are dense.
A second moment aggregated over $p^2$ and a mean aggregated over $p$ do not
subtract to a variance over anything. Production fixes this by importance
reweighting to `target='full'` ($q = p$), so both live on the actual viewing
distribution:

$$
\widehat{\mathbb{E}_q[g]} \;=\;
\frac{\sum_{\text{pairs}} \frac{q(e)}{p_{\text{pair}}(e)}\, g}
     {\sum_{\text{pairs}} \frac{q(e)}{p_{\text{pair}}(e)}},
\qquad
\frac{q}{p_{\text{pair}}} \;=\; \frac{\hat p(\rho_{ij})}{\hat p_{\text{pair}}(\rho_{ij})},
\tag{11}
$$

where each trajectory is reduced to a representative point $\rho$ by geometric
median (robust to a microsaccade inside the window), $\hat p$ is a Gaussian KDE
on those points, and $\hat p_{\text{pair}}$ is a *directly estimated* KDE on the
realized close-pair midpoints rather than the $p^2$ idealization
(`closepair_density='direct'`; see `../methods_eyepos_matching/note_closepair_density.md`).
Weights are self-normalized, so the unknown normalizer cancels; they are clipped
at $10^6 \times$ their median. Time bins enter with `pair_count` weighting,
i.e. pairs are simply pooled.

**The production form is uncentred.** `decompose.py:76` overrides the centred
estimator with

$$
\widehat{C_{\text{rate}}} \;=\; \underbrace{\tfrac12\big(P + P^\top\big)}_{\textstyle P \,=\, \sum_{\text{pairs}} \pi_{ij}\, Y^i (Y^j)^\top}
\;-\; \bar{Y}\,\bar{Y}^\top,
\qquad \sum_{ij}\pi_{ij} = 1,
\tag{12}
$$

with $\bar Y = $ `Erate`, the sample-weighted mean under the same target. This
is the legacy / §4.5-reference form. Its diagonal is exactly the classic
"second moment minus squared mean" and is **unconstrained in sign**: nothing
stops the estimate from going negative when the rate variance is small relative
to its own sampling error. That is the origin of the undefined cells in §6.1,
and `ceiling.py:63` deliberately returns `NaN` rather than clipping.

Note the two terms of (12) carry different weightings — the pair term is
importance-weighted at close-pair midpoints, the mean term is a sample-weighted
mean over all samples. Both target $p$, which is the point of the `full`
correction, but they are two different estimators of two different functionals
of the same distribution, so their errors do not cancel.

## 3.4 The ceiling table

`ceiling.py` reads the stage-1 record and, per unit per window, stores
`c_tot`, `c_rate`, `c_psth`, and the three ratios (`ceiling.py:61`). Undefined,
never clipped: `r2_max = NaN` unless both `c_tot > 0` and `c_rate > 0`.
`p_rate` is the fraction of the 1000 eye-shuffle nulls whose rate variance
reaches the real one.

Whole table: 2229 units, of which **19.3 % have $\hat c_{\text{rate}} \le 0$ at
`window_bins = 1`** and are undefined. Restricted to the figure 2 population the
attrition almost vanishes (22 of 994, 2.2 %), because rate $>2$ Hz and
split-half PSTH $R^2 > 0.10$ already select units with resolvable structure.

## 3.5 The window is not a free parameter

$R^2_{\max}$ rises steeply with the counting window, because summing counts
averages private noise down faster than it averages rate variance:

| `window_bins` | window | median $R^2_{\max}$ (population) | median $r^2_{\text{full}}/R^2_{\max}$ |
|---|---|---|---|
| 1 | 8.3 ms | 0.1235 | **0.350** |
| 2 | 16.7 ms | 0.2159 | 0.206 |
| 3 | 25 ms | 0.2885 | 0.160 |
| 6 | 50 ms | 0.3224 | 0.135 |

(The right column is deliberately wrong for rows 2–6: the $r^2$ is a per-bin
quantity, so only `window_bins = 1` is a legitimate normalizer. The column
exists to show the size of the error a mismatch would make — a factor of 2.6
between the first and last row.) `PRODUCTION_WINDOW_BINS = 1` is pinned in
`ceiling.py:56` and used in `_attach_ceiling` (`_ext_data.py:470`).

> Bookkeeping: the folder README quotes median $R^2_{\max} = 0.119$ at 1 bin and
> $0.279$ at 3. I measure $0.1235$ and $0.2885$ on the current caches with the
> same 972-unit population (every other count in the README — 22 dropped of 994,
> 294 of 972 above the shuffle threshold — reproduces exactly). The medians look
> like they predate a cache rebuild. Worth re-running the README numbers before
> anything is quoted in a caption.

---

# 4. The numerator: the single-trial $r^2$

## 4.1 What the code computes

Per condition, the twin's raw output `rhat` is passed through a per-unit affine
rescaling fit by Poisson MLE on the same data
(`rescale_rhat(..., mode='affine')`, `eval/eval_stack_utils.py:1192`):

$$
\hat y_i \;=\; e^{g}\,\hat r_i \;+\; e^{b},
\qquad
(g, b) = \arg\min \sum_{i \in \text{valid}} \big[\hat y_i - y_i \log \hat y_i\big].
\tag{13}
$$

The exponential parameterization makes $\hat y$ strictly positive by
construction, which is why the leave-one-out PSTH can go through the identical
transform without $\log 0$ blowing up the bits/spike row. Then
(`_ext_data.py:112`)

$$
r^2 \;=\; 1 - \frac{\operatorname{Var}\!\big[\hat y - y\big]}{\operatorname{Var}[y]},
\tag{14}
$$

with both variances taken over the pooled (trial $\times$ bin) axis, restricted
to bins with `dfs > 0`.

## 4.2 Three ways this differs from the idealized $r^2$ in (7) — and how much

**(i) Variance, not mean square.** (14) uses $\operatorname{Var}[\hat y - y]$
where (7) needs $\mathbb{E}[(\hat y - y)^2]$; the two differ by
$\big(\mathbb{E}[\hat y - y]\big)^2/\operatorname{Var}[y]$, so a biased
prediction is not penalized. And the bias is not forced to zero: the
stationarity conditions of (13) are $\sum_i y_i/\hat y_i = N$ and
$\sum_i (1 - y_i/\hat y_i)\hat r_i = 0$, neither of which is
$\sum_i (y_i - \hat y_i) = 0$.

*Measured:* I refit (13) offline on the rescaled leave-one-out PSTH for four
sessions. Median $\lvert\mathbb{E}[\hat y - y]\rvert / \mathrm{sd}(y)$ is
0.0002–0.0022, so the inflation of $r^2$ is $\le 5\times 10^{-6}$ — five orders
of magnitude below the $r^2$ values on the figure, and it prints as
`+0.00000` at five decimals. **This caveat is real in principle and dead in
practice.** (Measured on the PSTH predictor; the model conditions use the same
rescaling family, so I expect the same, but the near-constant `stab_global`
prediction is the one case I would spot-check if it ever mattered.)

**(ii) In-sample calibration.** $(g,b)$ are two parameters fit on the same
$\sim 10^5$ bins that are then scored. The optimism is $O(2/N)$ and negligible,
but it is shared identically across conditions in any case.

**(iii) The rescaling is fit on Poisson deviance, not squared error.** So $\hat y$
is not the $L_2$-optimal affine rescaling of `rhat`, and the reported $r^2$ is
slightly below what the same prediction could achieve. This is conservative and
uniform across conditions.

---

# 5. Does the ratio actually divide like quantities? Four checks

The bound (7) is exact for one distribution. The two halves of (1) come from two
pipelines. These are the checks that the division means something.

## 5.1 The PSTH cross-check (the strong one)

The decomposition independently estimates the ideal trial-average share,
$\operatorname{diag}C_{\text{psth}}/\operatorname{diag}C_{\text{total}}$. The
figure independently *measures* the $r^2$ of an actual leave-one-out PSTH
predictor. If the two pipelines' variances are commensurate, these must agree up
to the leave-one-out estimation noise.

Across the 972-unit population: correlation **0.959**, medians 0.0396
(decomposition) vs 0.0216 (measured), ratio 0.555.

The gap is expected and has the right size. A leave-one-out PSTH built from
$n-1$ other trials is $\hat\mu = \mu + \epsilon$ with
$\operatorname{Var}(\epsilon) \approx \operatorname{Var}(y\mid t)/(n-1)$, so

$$
r^2_{\text{LOO-PSTH}} \;\approx\;
\frac{\operatorname{diag}C_{\text{psth}}}{\operatorname{diag}C_{\text{total}}}
- \frac{1}{n-1}\cdot
\frac{\operatorname{diag}C_{\text{total}} - \operatorname{diag}C_{\text{psth}}}{\operatorname{diag}C_{\text{total}}}.
\tag{15}
$$

Median observed deficit 0.0176; median deficit predicted by (15) 0.0121 — same
order, observed $1.4\times$ predicted, which is what one expects since the
effective trial count per time bin is below the session trial count (bins are
dropped by the fixation mask), and since the two sides sit on different fixation
radii. The per-unit correlation of observed to predicted deficit is $\approx 0$,
i.e. (15) explains the *level* but not the unit-to-unit scatter, which is
dominated by estimator noise in $C_{\text{psth}}$.

**Read this as:** the two pipelines measure the same variance to within a
correction that is understood and small. That is the licence to divide.

## 5.2 Denominator commensurability

Two separate questions.

*Does the strided-window $C_{\text{total}}$ equal the plain per-bin variance?*
Comparing `c_tot` at `window_bins = 1` against $\operatorname{nanvar}$ of the
same aligned counts over all valid bins: median ratio **0.9971**, IQR
$[0.985,\,1.008]$, across 1585 units. Yes.

*Does the $0.5^\circ$ mask (denominator) match the $1.0^\circ$ mask
(numerator)?* Re-running the alignment at both radii:

| session | valid bins $0.5^\circ$ | at $1.0^\circ$ | $\operatorname{Var}(y)$ ratio | mean$(y)$ ratio |
|---|---|---|---|---|
| Allen_2022-02-16 | 5074 | 5739 (+13.1 %) | 0.964 | 0.959 |
| Logan_2020-01-07 | 4427 | 4787 (+8.1 %) | 0.985 | 0.985 |

The wider mask adds 8–13 % more bins, at lower firing rate, so it *lowers*
$\operatorname{Var}(y)$ by 1.5–3.6 %. Since $\operatorname{Var}(\lambda)$ falls
for the same reason, the net effect on the ratio is smaller still. Not zero, but
an order of magnitude below the effects that matter. If you want it exactly
right, the cheap fix is to move the decomposition to $1.0^\circ$ (or the figure
to $0.5^\circ$) rather than to correct after the fact.

## 5.3 Stimulus repetition and held-out status

Verified in §1.2: one distinct RSVP sequence per session, and fixRSVP absent
from the training `types`. Both are premises of the interpretation, not of the
algebra, and both hold.

## 5.4 Does normalizing change any conclusion?

Within a unit the normalization is a positive rescaling, so it cannot reorder
conditions: the sign of $r^2_{\text{full}} - r^2_{\text{stab\_global}}$ is
preserved for **100.0 %** of units (units with a negative denominator, which
would flip signs, are excluded by construction). What changes is the *pooled*
distribution, because units are no longer pooled on incommensurate scales. For
the well-behaved conditions the relative spread narrows:

| condition | IQR/median, raw | IQR/median, normalized |
|---|---|---|
| psth | 1.58 | 1.09 |
| full | 1.30 | 0.95 |
| ablated | 1.41 | 1.01 |
| stab_window | 1.33 | 0.97 |
| stab_trial | 1.80 | 1.41 |
| stab_global | 7.84 | 8.54 |

For conditions whose median sits near zero the ratio is meaningless in both
columns and the comparison should be ignored.

---

# 6. Pain points, ranked by how much they can move the number

## 6.1 The right tail (the one I would fix)

Dividing by a noisy, small, unconstrained denominator produces a heavy tail.
Population percentiles of $r^2_{\text{full}}/R^2_{\max}$:

$$
[1, 5, 25, 50, 75, 95, 99]\% \;=\;
[-0.45,\ -0.00,\ 0.20,\ 0.35,\ 0.53,\ 1.13,\ 2.71],
\qquad \max = 45.7 .
$$

The maximum is 45.7. Any mean over this is meaningless; medians and boxes are
fine, which is what the figure uses. But a single-unit gate cleans it up almost
for free, because the *median is remarkably insensitive to it*:

| gate | $N$ | median | IQR | 99th pct |
|---|---|---|---|---|
| none | 972 | 0.350 | $[0.201, 0.533]$ | 2.71 |
| $R^2_{\max} > 0.02$ | 940 | 0.344 | $[0.200, 0.518]$ | 1.37 |
| $R^2_{\max} > 0.05$ | 846 | 0.331 | $[0.197, 0.490]$ | 1.14 |
| $R^2_{\max} > 0.10$ | 598 | 0.325 | $[0.195, 0.455]$ | 0.96 |

Dropping 32 units with $R^2_{\max} < 0.02$ halves the 99th percentile and moves
the median by 0.006. **Recommendation: gate at $R^2_{\max} > 0.02$ and say so**,
or equivalently report the ratio only where the denominator is resolvable. It is
not a clip (the folder's undefined-not-clipped principle survives), it is an
inclusion rule on the denominator's estimability, exactly analogous to the
non-positive-$c_{\text{rate}}$ rule already in `ceiling.py`.

## 6.2 Negative values on an axis labelled "fraction"

42 % of units have a negative normalized $r^2$ under `stab_global`, 57 % under
`stab_global_ablated`. That is correct — those conditions predict worse than the
unit's own mean — but "fraction of explainable variance $= -0.02$" reads badly.
On figure 3 panel D, where `stabilized` is one of three boxes, the median
normalized value is $+0.024$ with a box straddling zero. The raw-$r^2$ panel has
the identical problem ($-0.0087$ to $+0.0138$ IQR) and currently solves it with
a dotted zero line; the same solution works here, but the *label* needs care.
"$r^2$ as a fraction of explainable variance" is honest; "fraction of
explainable variance captured" implies $[0,1]$ and is not.

## 6.3 $R^2_{\max}$ is a lower bound, and 6 % of units exceed it

§2.3(a). The reference line at 1 is an *expectation-of-an-oracle*, not a hard
bound: 6.1 % of units sit above it under `full`. The current figure draws it
dashed and labels it "ceiling" — I would relabel it something like "oracle
(measured)" and put the lower-bound status in the caption, otherwise a reader
sees 6 % of the data above a line labelled ceiling and concludes something is
broken.

## 6.4 The panel inherits figure 2's estimator

This is the real cost of the upgrade and it is not technical. Today panel D's
$y$ axis depends only on the twin and the spike counts. Normalized, it depends
additionally on: the close-pair threshold ($0.05^\circ$), the KDE bandwidth
(Scott's rule) entering the $1/\hat p$ weights, the geometric-median trajectory
reduction, the `direct` close-pair density, the weight clip, and the uncentred
$MM - \bar Y\bar Y^\top$ form. Every one of those is defended in
`../methods_eyepos_matching/`, but a reviewer who disputes any of them now
disputes panel D as well as figure 2. That is a defensible trade — the two
figures *should* speak the same language — but it should be a conscious one.

## 6.5 The shuffle null is not a gate here, and the subsets differ

`p_rate` tests whether a unit has resolvable *FEM* modulation (the shuffle
permutes trajectories but leaves $T_{\text{idx}}$ alone, so a shuffled
$C_{\text{rate}}$ retains PSTH variance). Gating on it would restrict the row to
FEM-modulated units, which is not what "fraction of total explainable variance"
means, so the figure correctly does not. But the two subsets behave differently
and it is worth knowing:

- FEM-resolvable ($p_{\text{rate}} \le 0.05$), $N = 678$: median 0.321
- not resolvable, $N = 294$: median 0.452

The non-resolvable units score *higher*, because their explainable variance is
mostly stimulus-locked and the twin captures stimulus-locked variance well. If
this ever gets reported, report both.

## 6.6 Weighting asymmetry between numerator and denominator

$C_{\text{rate}}$ is importance-weighted to target $p$ and pair-count-weighted
across time bins; $C_{\text{total}}$ is the unweighted legacy sample covariance
(§3.2). They therefore describe slightly different mixtures over $(t, e)$. The
methods note's §1.5 table treats $C_{\text{total}}$'s implicit weighting as
$\propto n_t$ (trial count), which coincides with pair-count weighting only when
$n_t$ is constant across time bins — and it is not, since fixation durations
vary. The $r^2$'s own denominator is a third weighting (unweighted over all
valid bins). The empirical check in §5.2 says the *diagonal* consequence is
$\le 0.3$ %, which is reassuring, but the asymmetry is structural rather than
bounded and is worth one line in any methods text.

## 6.7 Subject asymmetry

| subject | $N$ | median $R^2_{\max}$ | median full | median stab_global |
|---|---|---|---|---|
| Allen | 835 | 0.130 | 0.357 | $+0.026$ |
| Logan | 137 | 0.094 | 0.292 | $-0.011$ |

The normalization does not create the asymmetry, and it does not remove it
either. Logan's lower ceiling and lower normalized score move together, so the
normalized axis is not obviously the more favourable one for the weaker subject.

## 6.8 Join fragility

Both pipelines key on the dataset's neuron column index. They reach that index
through different configs (`multi_basic_120_long.yaml` for the decomposition,
the checkpoint's own configs for the twin) and different spike thresholds. The
key is only valid if both see identical unit orderings. `tests/test_ceiling.py`
pins the *contract* (ids, not compacted indices) but nothing asserts agreement
between the two live pipelines. A one-line assertion — that the per-unit mean
rate implied by each pipeline agrees within a few percent for every joined key —
would close this permanently and is cheap. Today the evidence that the join is
correct is indirect but strong: the 0.959 correlation in §5.1 could not survive a
scrambled key.

---

# 7. Recommendation

**Promote it, with three changes.**

The case for: 0.0396 is uninterpretable on its own; 0.35 is a sentence
("the twin captures about a third of the variance that is capturable in
principle, and about as much as a noise-free trial average could — while
capturing it for a different reason, since it survives behavioral ablation and
collapses under stabilization"). The normalization cannot reward a condition for
predicting less, because the denominator is estimated from data alone and is
identical across all conditions — unlike the self-consistency row, whose
denominator moves with the condition. It cannot reorder conditions within a
unit. It ties panel D to figure 2's measurement, which is the paper's central
quantitative claim, so figure 3 stops being scale-free.

The changes:

1. **Gate on $R^2_{\max} > 0.02$** (§6.1). Costs 32 of 972 units and the median
   moves by 0.006; buys a 99th percentile of 1.37 instead of 2.71 and removes a
   45$\times$ outlier from a panel that would otherwise have one.
2. **Relabel the reference line.** It is an oracle estimated under coarser
   conditioning than the model enjoys, so it is a lower bound and 6 % of units
   sit above it (§2.3, §6.3). Do not call it a ceiling in the axis label.
3. **Say what the denominator is in the caption**, in one clause: *rate variance
   over total count variance, per unit, measured in Fig. 2 at the model's 120 Hz
   bin.* The window dependence (§3.5) is a factor of 2.6 across the windows
   figure 2 reports, so a reader who assumes 25 ms will misread the panel by
   more than the ablation effect.

Optionally: keep the raw $r^2$ as a supplementary panel or a second axis. It is
one line of code and it forecloses the reviewer question "what does this look
like before you divided by something you estimated".

What I would *not* do: gate on the eye-shuffle null (§6.5), clip the ratio to
$[0,1]$ (§6.2), or present a mean of the ratio anywhere (§6.1).

---

# Appendix: reproducing every number in this note

All figures quoted here come from the committed caches
(`outputs/cache/covdecomp_ceiling.pkl`, `outputs/cache/fig3e_extended_ablation.pkl`,
`outputs/cache/covdecomp_aligned_sessions.pkl`) plus two live re-derivations:

- **§1.2, stimulus repetition** — `_ext_stim.FixRsvpRenderer('Allen_2022-02-16')`,
  comparing `trial.image_ids[hist_idx]` and `trial.positions[hist_idx]` across
  trials aligned by within-trial sample index.
- **§4.2(i), the variance-vs-MSE inflation** — leave-one-out PSTH rebuilt from
  the aligned cache, passed through `rescale_rhat(mode='affine')`, then scoring
  $1 - \operatorname{Var}(\text{resid})/\operatorname{Var}(y)$ against
  $1 - \mathbb{E}[\text{resid}^2]/\operatorname{Var}(y)$.
- **§5.2, the fixation-radius comparison** — `align_fixrsvp_trials` run at
  `fixation_radius` 0.5 and 1.0 on two sessions.

The build for this note matches the folder convention:

```
pandoc note_r2_ceiling.md -s --mathml --self-contained \
    --lua-filter=../methods_eyepos_matching/number-eqs.lua \
    -o note_r2_ceiling.html
```
