---
title: "Extending McFarland's cross-trial decomposition to non-uniform trials and non-homogeneous stimuli"
subtitle: "A methodological note"
author: "fem-v1-fovea methods note"
date: "2026-05-29"
---

# Summary

McFarland, Cumming & Butts (2016) introduced a cross-trial,
eye-position-conditioned decomposition that separates the stimulus-driven and
stimulus-independent components of V1 responses in the presence of fixational
eye movements (FEMs). It is now the standard estimator for the FEM fraction
$1-\alpha$, the corrected noise covariance $C_{\text{noise}}^{\text{corr}}$, and the
Fano factor under FEMs. Two assumptions sit inside it:

- **(A1) Uniform trial/phase structure** — every stimulus phase $t$ has the
  same number of trials $n_t$, so the across-phase weighting is the same for
  every term in the law-of-total-(co)variance (LOTC).
- **(A2) Statistically stationary stimulus** — the across-phase distribution
  of $r(t,e)$ at a fixed eye position is the same at every $e$
  (equivalently, $\mathbb{E}_t[r^k(t,e)]$ is independent of $e$ for the
  moments used). It then does not matter *which* eye-position distribution
  the rate moments are averaged over, and the close-pair restriction in (7)
  is free.

Both assumptions fail in the structured `fixRSVP` stimulus used here.
Fixation durations vary across trials, so the per-phase trial count $n_t$ drops
across phases; and the windowed natural-image stimulus is far from
translation-invariant, so the rate $r(t,e)$ depends on absolute eye position.
This note develops, validates, and discusses a methodological extension of
McFarland et al. for each assumption violation:

1. **Consistent phase weighting** under variable $n_t$. The close-pair rate
   estimator is intrinsically pair-count weighted ($\propto n_t(n_t{-}1)/2$)
   across phases; for the LOTC to hold term-by-term, the PSTH covariance and
   the mean entering the rate-variance subtraction must use the same
   pair-count weighting. Validated against closed-form synthetic ground truth
   with variable per-phase trial counts.
2. **Eye-position-distribution matching** under a non-homogeneous stimulus.
   Close pairs at threshold $\Delta e<\varepsilon$ are sampled in proportion
   to the *squared* eye-position density $p(e)^2$, while the total covariance
   and the PSTH covariance are over $p(e)$. We restore consistency with an
   importance-reweighted estimator that has the target eye distribution as a
   parameter, with two principled choices ($p$ and $p^2$) whose gap is a
   model-free measure of stimulus non-homogeneity.

Sections 1 and 2 set up McFarland's estimator and the synthetic model that
breaks both assumptions. Sections 3 and 4 develop and validate one extension
each on the synthetic. Section 5 reports the consequences on synthetic and
real data. Section 6 records the production-pipeline state — Extension 1 has
already been integrated; Extension 2 is implemented in this folder and gated.

---

# 1. Background: the cross-trial decomposition under FEMs

## 1.1 The law of total (co)variance

Let $Y_c^i(t)$ be the spike count of neuron $c$ on trial $i$ at stimulus phase
$t$ (a frozen-stimulus time bin). Each trial samples an absolute eye position
$e$ from the fixational distribution $p(e)$; we treat $e$ as approximately
constant over the short counting window and approximately independent of $t$.
Writing the firing rate as $r_c(t,e) = \mathbb{E}[Y_c \mid t, e]$, the **law of
total variance** partitions the stimulus-driven rate variance into a
stimulus-locked (PSTH) part and an eye-movement (FEM) part:

$$
\underbrace{\mathrm{Var}_{t,\,e\sim p}\!\big[r_c(t,e)\big]}_{\text{total rate variance}}
 \;=\;
\underbrace{\mathrm{Var}_t\!\Big[\,\mathbb{E}_{e\sim p}\!\big[r_c \mid t\big]\Big]}_{\text{PSTH variance}}
 \;+\;
\underbrace{\mathbb{E}_t\!\Big[\,\mathrm{Var}_{e\sim p}\!\big[r_c \mid t\big]\Big]}_{\text{FEM variance}} .
\tag{1}
$$

McFarland et al. define the fraction of stimulus-driven variance captured by
the PSTH,

$$
\alpha_c \;=\; \frac{\mathrm{Var}_t\big[\mathbb{E}_{e\sim p}[r_c\mid t]\big]}
                     {\mathrm{Var}_{t,e\sim p}[r_c]} ,
\qquad
1-\alpha_c \;=\; \text{FEM fraction}.
\tag{2}
$$

The pairwise analogue partitions the count covariance,
$\mathrm{Cov}_{t,e\sim p}[Y_m,Y_n] = \mathrm{Cov}_{t,e\sim p}[r_m,r_n] + \mathbb{E}\big[\text{noise cov}\big]$,
so the FEM-corrected (stimulus-independent) covariance is

$$
C_{\text{noise}}^{\text{corr}} \;=\; C_{\text{total}} \;-\; C_{\text{rate}},
\qquad
\text{Fano}_c \;=\; \frac{\big(C_{\text{noise}}^{\text{corr}}\big)_{cc}}{\bar r_c},
\tag{3}
$$

where $C_{\text{total}} = \mathrm{Cov}_{t,e\sim p}[Y]$ is the raw count
covariance and $C_{\text{rate}} = \mathrm{Cov}_{t,e\sim p}[r]$ is the total
rate covariance over the **same distribution $p$**. Equation (3) is only
meaningful if $C_{\text{rate}}$ and $C_{\text{total}}$ are measured over one
and the same eye-position distribution; this is the consistency requirement
that the rest of this note is about.

## 1.2 The cross-trial trick: distinct trials cancel the observation noise

We never observe $r_c(t,e)$ directly — only the noisy counts $Y_c$. The engine
of the whole method is that the observation noise is independent across
**distinct** trials, so for two different trials $i\neq j$ at the same phase
$t$,

$$
\mathbb{E}\big[Y_c^i(t)\,Y_c^j(t)\,\big|\,e_i,e_j\big] = r_c(t,e_i)\,r_c(t,e_j).
\tag{4}
$$

The product of two distinct trials' counts is an unbiased estimate of the
product of their underlying rates, with the Poisson/observation variance
removed. Every estimator below is a different way of averaging these
distinct-trial products — and **the choice of which pairs to average sets the
eye-position distribution**.

## 1.3 Two families of estimator

**Signal (PSTH) variance and covariance — all distinct pairs.** Averaging (4)
over *all* distinct pairs at the same phase, then over phases, estimates the
PSTH (stimulus-locked) second moment (McFarland's Eq. 6):

$$
\widehat{\mathrm{Var}_t\!\big[\mathbb E_i\, r^i_c(t)\big]}
 = \big\langle \langle Y_c^i(t)\,Y_c^j(t)\rangle_{i\ne j}\big\rangle_t - \bar Y_c^{\,2},
\tag{5}
$$
$$
\widehat{\mathrm{Cov}_t\!\big[\mathbb E_i\, r^i_m,\ \mathbb E_i\, r^i_n\big]}
 = \big\langle \langle Y_m^i(t)\,Y_n^j(t)\rangle_{i\ne j}\big\rangle_t - \bar Y_m\,\bar Y_n.
\tag{6}
$$

Because *all* same-phase pairs are used, the phase mean averages over the
**full** fixational distribution $p(e)$: these terms live on $p(e)$.

**Eye-conditioned rate variance and covariance — close pairs only.** Restrict
the same averages to pairs whose eye trajectories nearly coincide,
$\Delta e_{ij}<\varepsilon$. This drives $e_i\approx e_j\approx e$, so
(4) $\to r_c(t,e)^2$: the *total* rate second moment with the Poisson noise
removed (their Eqs. 8 and 16),

$$
\widehat{\mathrm{Var}_{i,t}[r_c]}
 = \big\langle \langle Y_c^i(t)\,Y_c^j(t)\mid \Delta e_{ij}<\varepsilon\rangle_{i\ne j}\big\rangle_t - \bar Y_c^{\,2},
\tag{7}
$$
$$
\widehat{\mathrm{Cov}_{i,t}[r_m,r_n]}
 = \big\langle \langle Y_m^i(t)\,Y_n^j(t)\mid \Delta e_{ij}<\varepsilon\rangle_{i\ne j}\big\rangle_t - \bar Y_m\,\bar Y_n.
\tag{8}
$$

The close-pair restriction is exactly what cancels the Poisson noise across
distinct trials — but, as §4 shows, it also silently changes the eye-position
distribution these terms live on.

## 1.4 The reported quantities

McFarland's reported quantities combine the two estimator families. The FEM
fraction is $\alpha_c = (5)/(7)$ (Eq. 10); the corrected noise covariance and
Fano factor are $C_{\text{noise}}^{\text{corr}} = C_{\text{total}} - C_{\text{rate}}$
from (3) with the close-pair $C_{\text{rate}}$ from (8). McFarland et al. also
give an analytical form for $\alpha$ via the Fourier transform of $p$ (their
Eq. 9), in which $\alpha$ weights the rate spectrum $|R(k)|^2$ by the
eye-distribution spectrum $|P(k)|^2$.

## 1.5 Two assumptions, made explicit

Two assumptions are baked into the estimators above and into McFarland's
analytical $\alpha$:

- **(A1) Uniform trial/phase structure.** Equations (5)–(8) and the LOTC (1)
  presume that *the same across-phase weighting applies to every term*. The
  close-pair estimator (7) intrinsically uses pair-count weighting: each phase
  $t$ contributes $n_t(n_t{-}1)/2$ same-phase pairs. The PSTH estimator (5)
  is, in McFarland's formulation, also pair-count weighted; in implementation
  it is sometimes computed as a split-half cross-covariance with a different
  across-phase weighting (e.g. uniform $1/T$). When $n_t$ is constant the
  weightings agree; when $n_t$ varies they need not, and the LOTC fails
  term-by-term.
- **(A2) Statistically stationary stimulus.** The across-phase distribution
  of $r(t,e)$ at a fixed eye position is the same at every $e$, i.e.
  $\mathbb{E}_t[r^k(t,e)]$ is independent of $e$ for the moments used. Then
  $\mathbb{E}_{e\sim D}\!\big[\mathbb E_t[r^k(t,e)]\big]$ is the same for
  every distribution $D$ over $e$, so the close-pair restriction in (7)–(8)
  — which silently sets $D=p^2$ — gives the same answer as the actual
  viewing target $D=p$. McFarland et al. state the assumption at the level
  of the stimulus (their text around Eqs. M7–M10):

  > "by restricting analysis to trial pairs where $\Delta e_{ij}\approx 0$,
  > [the estimator] gives an estimate of $\mathbb{E}[r^2(e,t)]$ under the
  > conditional eye position distribution $p(e\mid\Delta e\approx 0)$ rather
  > than $p(e)$. However, for a stimulus that is statistically invariant to
  > spatial translations (such as used in our study), the expectation of
  > $r^2(e,t)$ with respect to any distribution over $e$ will be the same."

  Stimulus stationarity is a sufficient condition: when each frame is a
  sample from a translation-invariant random field, the rate distribution
  at one eye position is, in distribution, the rate distribution at any
  other, so the across-phase moments at fixed $e$ do not depend on $e$.
  McFarland's ternary bar noise has this property; the structured `fixRSVP`
  images used here do not.

Both assumptions fail in the structured `fixRSVP` paradigm used here. §2
states the violations and the synthetic model that captures them; §3 and §4
develop a methodological extension for each.

---

# 2. Our setting and the synthetic model

## 2.1 Violation of (A1): uneven fixation durations $\Rightarrow$ variable $n_t$

`fixRSVP` trials are organized around fixational fixation periods of variable
duration. Aligned to fixation onset and binned into stimulus phases, this
gives a monotonically decreasing per-phase trial count: every trial is
fixating in early bins, but only long-fixation trials contribute to late bins.
On a representative session (Allen 2022-04-13, 49 cells) $n_t$ ranged from 42
to 78 across phases (CV $\approx 0.19$); on our broader set of cells the
spread is larger. Under this $n_t$ variation, the close-pair rate estimator
weights each phase by $n_t(n_t{-}1)/2$ pairs while a uniform split-half PSTH
estimator weights each phase by $1/T$ — these only coincide if $n_t$ is
constant.

## 2.2 Violation of (A2): the windowed `fixRSVP` stimulus

`fixRSVP` presents a natural image at a fixed location on the screen, behind a
spatial **window** (aperture). The neuron's receptive field (RF) is fixed in
retinotopic coordinates, so in screen coordinates it moves with the eye. Two
mechanisms make the **across-phase distribution** of $r_c(t,e)$ at fixed eye
position depend on absolute eye position — directly violating (A2):

1. **Windowing — a hard non-stationarity.** Once the fixation offset is large
   enough to carry the RF off the windowed image, the RF samples only the
   uniform gray background and the rate collapses to the gray-screen baseline
   **regardless of stimulus phase $t$**. So at peripheral $e$ the across-phase
   distribution of $r$ is concentrated at baseline; at central $e$ it is
   strongly stimulus-modulated. The across-phase moments at fixed $e$ are
   therefore strongly $e$-dependent — *before any image content is
   considered*.
2. **Image structure.** Within the window, drift slides the structured image
   across the RF, so which feature drives the cell — and how strongly —
   depends on absolute eye position. The across-phase distribution of $r$ at
   each $e$ samples a different patch of a fixed image, not different draws
   from a translation-invariant ensemble; the per-$e$ moments differ
   accordingly.

## 2.3 A minimal generative model that breaks both assumptions

We validate every estimator claim below against a synthetic generator
(`synthetic.py`) chosen as the *smallest* abstraction that captures both
violations and admits a closed-form LOTC decomposition. The reader should be
able to evaluate whether each ingredient is a faithful minimal model of the
corresponding real-data feature before reading the estimator validations in
§3 and §4.

**Rate field.** For neuron $c$, stimulus phase $t$, and absolute eye position
$e\in\mathbb R^2$,

$$
r_c(t,e) \;=\; \mathrm{base} \;+\; a\,\bar P_c(t)\,\eta(t) \;+\; b\,F_c(e),
\tag{9}
$$

with three deliberately separable ingredients:

- $\bar P_c(t)$ — a zero-mean phase-locked drive (the stimulus-locked "PSTH").
  This is the part of the rate that the cross-trial trick recovers via
  same-phase pairing (5).
- $\eta(t)$ — an optional per-phase amplitude envelope. By default $\eta\equiv 1$.
  When the synthetic targets Extension 1 (§3), we set $\eta(t)$ to decay across
  phases. This mirrors `fixRSVP`'s onset transients, which concentrate
  stimulus-locked amplitude in early — and high-$n_t$ — phases; without that
  correlation, pair-count and uniform phase weights give the same
  $\mathrm{Var}_t[\bar P]$ in expectation and the Extension-1 bias washes out.
- $F_c(e)$ — the **eye-position-sensitivity profile**. A constant $F$ recovers
  McFarland's homogeneous stimulus; any non-constant $F$ breaks (A2).

Four profile shapes span the relevant cases:

| profile | $F_c(e)$ | abstracts |
|---|---|---|
| `flat` | constant | the homogeneous stimulus — McFarland's working regime |
| `central` | Gaussian peaked at $e=0$ | a windowed stimulus that drives the cell strongly near fixation and weakly in the periphery |
| `eccentric` | $\propto \lVert e\rVert^2$ | the *complement* — a cell that becomes more sensitive as the eye moves off center |
| `linear` | $\propto x$ | the simplest spatial gradient |

`central` is the workhorse stand-in for the windowed `fixRSVP` regime. `flat`
isolates Extension 1: with no eye dependence the truth $1-\alpha=0$ is
independent of the eye-position distribution, so any deviation is purely a
phase-weighting artifact. `eccentric` and `linear` stress-test the corrected
estimator on profiles whose variance lives in the periphery, where close
pairs are rarest.

**Eye distribution.** Eyes are drawn i.i.d. $e\sim p=\mathcal N(0,\sigma^2 I)$
per (trial, phase), $\sigma=0.15^\circ$ (realistic fixational drift). This
choice is deliberate: for an isotropic Gaussian $p$, the close-pair density
$p(e)^2$ is *exactly* $\mathcal N(0,\sigma^2/2\,I)$ — a tighter Gaussian with
half the variance — so the LOTC under either $p$ or $p^2$ has a closed form
by direct sampling (§A.1).

**Trial structure.** The (trial, phase) array has shape $(N_{\text{tr}}, T)$.
With $n_{\text{tr/phase}}=\text{None}$ every phase has $N_{\text{tr}}$
trials, recovering McFarland's uniform-trial regime. To violate (A1) we set
$n_{\text{tr/phase}}=(n_t)_{t=1}^{T}$ — a monotonically decaying staircase
(default lo $15$, hi $\sim N_{\text{tr}}$, see Fig. 1A) that mimics the
fixation-duration distribution. Entries beyond $n_t$ in each phase are
masked invalid.

**Observation noise.** Spikes are drawn
$Y_c \sim \mathrm{Poisson}(r_c(t,e))$ (optionally with a shared latent giving
a known stimulus-independent noise covariance). Independent Poisson noise
across distinct trials is exactly the property the close-pair estimator
exploits.

**Closed-form ground truth.** Because $F$ depends only on $e$ and $\bar P$
only on $t$, the LOTC decomposition admits a closed form under any
distribution $D$ over $e$ and any phase weighting $w_t$:

$$
\mathrm{Var}_{\text{psth}}^c(w) = a^2\,\mathrm{Var}_w\!\big[\bar P_c(t)\,\eta(t)\big],
\qquad
\mathrm{Var}_{\text{fem}}^c(D) = b^2\,\mathrm{Var}_{e\sim D}\![F_c(e)],
\tag{10}
$$

and $(1-\alpha)^c(D,w) = \mathrm{Var}_{\text{fem}}^c(D) /
(\mathrm{Var}_{\text{psth}}^c(w) + \mathrm{Var}_{\text{fem}}^c(D))$. The
matched-estimator targets for §3 are the pair-count phase weighting
$w_t\propto n_t(n_t{-}1)/2$ on the full distribution $D=p$; for §4 the targets
are $D\in\{p, p^2\}$ with pair-count $w_t$ throughout. Both are obtainable to
arbitrary precision by direct sampling (`synthetic.ground_truth`).

This is the model used throughout. It is the minimal one for which McFarland's
LOTC has a closed form, both assumptions can be broken independently, and
each fix can be verified against truth.

---

# 3. Extension 1: consistent phase weighting under variable $n_t$

## 3.1 The phase-weighting mismatch, derived from the close-pair estimator

The close-pair rate estimator (7) averages products $Y_c^i(t)\,Y_c^j(t)$ over
distinct same-phase pairs at $\Delta e_{ij}<\varepsilon$. With $n_t$ trials at
phase $t$, the number of available pairs is $n_t(n_t{-}1)/2$. Averaging
uniformly over pairs is therefore averaging over phases with the **pair-count
weight**

$$
w_t \;\propto\; n_t(n_t-1)/2.
\tag{11}
$$

Under constant $n_t$ this is a global constant and the choice is invisible;
under variable $n_t$ it concentrates weight on high-$n_t$ phases. **This is
the phase weighting at which the close-pair second moment in (7) lives.** For
the LOTC (1) to hold term-by-term the PSTH variance (5) and the mean $\bar Y$
that enters (7) must use the same $w_t$.

Two historical mismatches against (11) are the focus here:

- **Mismatch 1 (mean in $C_{\text{rate}}$).** A trial-count-weighted global
  mean $\bar Y = \mathrm{mean}_i Y_c^i$ weights phases by $n_t$, not by
  $n_t(n_t{-}1)/2$. Then
  $C_{\text{rate}} = \mathbb E_{\text{pair}}[YY^\top] - \bar Y\,\bar Y^\top$
  is not a covariance under any single phase weighting.
- **Mismatch 2 (PSTH estimator).** A split-half PSTH cross-covariance with
  uniform across-phase weighting (every phase weighted $1/T$) estimates
  $\mathrm{Cov}_{w_{\text{uni}}}[\bar P(t)]$, while $C_{\text{rate}}$ estimates
  $\mathrm{Var}_{w_{\text{pair}}}[\bar P(t)]$. Even when both target only the
  PSTH (i.e. under a homogeneous stimulus, where the close-pair restriction
  is free), they differ by the asymmetric-amplitude bias
  $\mathrm{Var}_{w_{\text{pair}}}[\bar P] - \mathrm{Var}_{w_{\text{uni}}}[\bar P]$.

The fix is to use $w_t = n_t(n_t{-}1)/2$ for $\bar Y$ and for the PSTH
cross-covariance, matching the close-pair estimator's intrinsic weighting.

## 3.2 The shuffle null is positive under the mismatch

A useful diagnostic: under a *shuffle null* that destroys the eye–spike
coupling (random eye trajectories), the close-pair estimator has no
real eye dependence to exploit and $C_{\text{rate}}^{\text{shuf}}$ should
converge to the (pair-count-weighted) PSTH covariance. If $C_{\text{psth}}$
is computed with a *different* weighting, then $D_z = f_z(C_{\text{noise}}^{\text{corr}})
- f_z(C_{\text{noise}}^{\text{uncorr}})$ deviates from zero under the null *purely* because
of the weighting mismatch, with no eye-position dependence on which the (A2)
violation could act. This is exactly the shuffle-null bias diagnosed and
fixed in `ryan/fig2/bias_diagnosis/` ($D_z^{\text{shuf}}: -0.0068
\to +0.0010$, $p$: $<10^{-4}\to 0.44$, after consistent pair-count weighting).

## 3.3 Validation on the synthetic with variable $n_t$ and a `flat` profile

Synthetic ground truth makes the same point in closed form. Take the `flat`
profile, so $F=0$ and the truth is $1-\alpha=0$ under every eye distribution
and every phase weighting (Var$_{\text{fem}}=0$). Pair a staircase $n_t$
($15\to360$ across $T=100$ phases) with an envelope $\eta(t)$ that
concentrates PSTH amplitude in early, high-$n_t$ phases (Fig. 1A); this
matches `fixRSVP`'s onset transients. Apply `decompose` with the matched
(`phase_weighting='pair_count'`) and unmatched (`phase_weighting='uniform'`)
variants to deterministic rates (so any deviation is a weighting artifact,
not Poisson sampling).

![**Figure 1 — Consistent phase weighting validated against synthetic ground
truth.** **(A)** The variable-$n_t$ staircase across phases and the two
phase-weight curves: pair-count $w_t\propto n_t(n_t{-}1)/2$ (matched, blue)
and uniform $w_t=1/T$ (unmatched, red). Under constant $n_t$ both are flat;
under variable $n_t$ the pair-count weight strongly emphasizes early
high-$n_t$ phases. **(B)** Homogeneous (`flat`) profile + envelope: the truth
is $1-\alpha=0$ for every weighting. The unmatched (uniform) phase weighting
biases $1-\alpha$ well above 0 (median 0.39); the matched (pair-count)
weighting recovers the truth (median $\sim 0$). **(C)** Structured profiles
(`central`, `eccentric`, `linear`) under the same staircase + envelope: the
matched estimator (○) lies on the pair-count-weighted ground-truth identity
line for the actual viewing distribution $p$; the unmatched (×) is off in a
profile-dependent way.](figures/fig_phase_weighting.png)

Panel B is the synthetic analogue of the bias_diagnosis shuffle-null bias:
the eye trajectories are real, the rate has no eye dependence, so the bias is
*purely* the Cpsth/Crate phase-weighting mismatch. Panel C adds a spatial
profile and confirms that the matched estimator tracks the ground-truth
$1-\alpha$ under the actual viewing distribution while the unmatched is
biased in a profile-dependent way.

## 3.4 What the correction does and does not address

Extension 1 is *only* about the weighting *across* phases. It is a no-op for
constant $n_t$ and operates at the level of the LOTC's term-by-term
consistency. It does not address Extension 2's eye-position distribution
mismatch, which acts *within* phases on the choice of $e$ used to average the
rate variance. The two are orthogonal: Extension 1 fixes the consistency of
the LOTC under variable $n_t$; Extension 2 fixes the consistency of the LOTC
under a non-homogeneous stimulus.

---

# 4. Extension 2: eye-position-distribution matching under a non-homogeneous stimulus

## 4.1 Close pairs are sampled from $p(e)^2$

Take two trials with eye positions $e_i, e_j$ drawn independently from $p$.
The density of close pairs *at position $e$* — pairs with $e_i\approx e_j
\approx e$ within the threshold ball — is proportional to $p(e)\cdot p(e) =
p(e)^2$. Hence the close-pair estimator (7) does not measure
$\mathbb{E}_{e\sim p}[r^2]$ but

$$
\big\langle Y_c^i Y_c^j \mid \Delta e<\varepsilon\big\rangle
 \;\;\longrightarrow\;\;
\mathbb{E}_{e\sim p^2}\!\big[r_c(t,e)^2\big]
\quad (\varepsilon\to0),
\qquad
p^2(e) \equiv \frac{p(e)^2}{\int p(e')^2\,de'} .
\tag{12}
$$

For an isotropic Gaussian fixation $p=\mathcal N(0,\sigma^2 I)$ the close-pair
distribution is *exactly* $p^2=\mathcal N\!\big(0,\tfrac{\sigma^2}{2}I\big)$:
a tighter, more central Gaussian with **half the variance**. Figure 2
confirms this geometrically and numerically and shows how
$\mathrm{Var}_{p}[F]$ and $\mathrm{Var}_{p^2}[F]$ differ across profiles.

![**Figure 2 — Close pairs sample the squared density $p(e)^2$.** **(A)** Eye
positions (grey) and the representative positions of distinct-trial close
pairs (red) for $\sigma=0.15^\circ$, threshold $0.05^\circ$; the $1,2\sigma$
circles of $p$ (solid) enclose the tighter $1,2\sigma/\sqrt2$ circles of $p^2$
(dashed). **(B)** The $x$-marginal: the close-pair distribution matches
$\mathcal N(0,\sigma^2/2)$ (observed variance ratio $\approx 0.5$). **(C)**
Consequently the FEM variance $\mathrm{Var}[F]$ of an eye-sensitivity profile
differs between $p$ and $p^2$, and the sign of the resulting $1-\alpha$ bias
depends on the profile.](figures/fig_mechanism.png)

## 4.2 The decomposition is consistent only on one distribution

Combining §1 and §4.1: in the non-homogeneous case the *naive* pipeline
measures

$$
C_{\text{total}},\,C_{\text{psth}} \ \text{over } p(e),
\qquad
C_{\text{rate}} \ \text{over } p(e)^2 ,
$$

so the derived quantities mix two distributions,

$$
1-\alpha = 1 - \frac{C_{\text{psth}}(p)}{C_{\text{rate}}(p^2)},
\qquad
C_{\text{noise}}^{\text{corr}} = C_{\text{total}}(p) - C_{\text{rate}}(p^2).
\tag{13}
$$

The LOTC (1) holds term-by-term **only when every term is taken over the same
distribution**. Equation (13) violates this whenever
$\mathbb{E}_{p^2}[r^2]\neq\mathbb{E}_p[r^2]$, i.e. whenever the rate depends
on absolute eye position — precisely the non-homogeneous case.

## 4.3 A sharper inconsistency in the naive estimator

There is a second, subtler defect already present in (7) as written. The
close-pair second moment is over $p^2$, but the term subtracted from it —
$\bar Y_c^2$, the global mean — is over $p$:

$$
\big(C_{\text{rate}}\big)_{cc}^{\text{naive}}
 = \underbrace{\mathbb{E}_{p^2}[r_c^2]}_{\text{2nd moment over }p^2}
 - \underbrace{\big(\mathbb{E}_{p}[r_c]\big)^2}_{\text{mean over }p}.
\tag{14}
$$

This is **not a variance under any single distribution**. Because
$\mathbb{E}_{p^2}[r]>\mathbb{E}_p[r]$ for a centrally-peaked profile and
$\mathbb{E}_{p^2}[r]<\mathbb{E}_p[r]$ for an eccentric one, the naive
$C_{\text{rate}}$ can be *inflated above* the true $p^2$ variance (central
cells) or driven *negative* (eccentric cells). We confirm both below.

Note that this is a different axis from Extension 1: Extension 1's
shuffle null is **blind** to (A2) because shuffling eye trajectories
across trials destroys the eye–spike coupling, removing exactly the
$r(e)$ dependence on which the distribution mismatch acts. A
shuffle-null $D_z\approx 0$ therefore says nothing about eye-position
consistency.

## 4.4 The corrected estimator: target distribution as a parameter

The close-pair conditioning is *required* to cancel the independent Poisson
noise, so we cannot avoid sampling from $p^2$ when forming the second-moment
estimate. But the close pairs give Poisson-free **local** estimates of
$r(t,e)^2$; the bias lives entirely in how those local estimates are
**aggregated over $e$**. We re-weight toward a chosen target distribution
$q(e)$ by importance sampling. A sample drawn from $p$ contributes weight
$q(e)/p(e)$; a close *pair* drawn from $p^2$ contributes weight
$q(e)/p(e)^2$:

$$
\widehat{\mathbb{E}_q[g]}
 = \frac{\sum_{\text{pairs}} \tfrac{q(e)}{p(e)^2}\, g}{\sum_{\text{pairs}} \tfrac{q(e)}{p(e)^2}}
 \quad\text{(second moment)},
\qquad
\widehat{\mathbb{E}_q[h]}
 = \frac{\sum_{\text{samples}} \tfrac{q(e)}{p(e)}\, h}{\sum_{\text{samples}} \tfrac{q(e)}{p(e)}}
 \quad\text{(mean, total, PSTH)} .
\tag{15}
$$

Two choices of $q$ make all terms consistent (module `estimators.py`,
`target=`):

| target | $q$ | close-pair weight $q/p^2$ | sample weight $q/p$ | character |
|---|---|---|---|---|
| **Direction 1** (`full`) | $p$ | $1/p$ (unbounded) | $1$ | actual viewing distribution |
| **Direction 2** (`central`) | $p^2$ | $1$ | $\propto p$ (bounded) | central; close-pair-supported |
| (naive) | — | $1$ on $p^2$ | $1$ (mean over $p$) | inconsistent (Eq. 14) |

Both directions fix the mean/second-moment inconsistency of (14) by
construction. They are **the two consistent resolutions** of the mismatch:
push everything out to the full distribution, or pull everything in to the
close-pair distribution.

**Equivalence to eye-position stratification.** Importance reweighting (15) is
equivalent to **stratifying by absolute eye position**: partition $e$ into
strata, estimate the close-pair second moment *within* each stratum (where
$p\approx$ const so $p^2\approx p$ locally and the local estimate is unbiased
for $\mathbb E[r^2\mid s]$), then aggregate strata with weights equal to the
target occupancy $q(s)$. As the strata shrink this is exactly (15); with
finite strata it pools pairs before weighting, trading a little resolution for
lower variance. This is the natural generalization of McFarland et al.: their
estimator conditions on phase $t$ and on $\Delta e\approx0$; the correction
conditions *additionally* on absolute eye position.

## 4.5 Direction 1 vs Direction 2: tradeoff and non-homogeneity diagnostic

The two consistent directions are not equally easy to estimate. Direction 1
(target $p$) requires the unbounded weight $1/p$, largest in the periphery —
exactly where close pairs are rarest. For an eccentric-sensitive cell whose
variance lives in the periphery, this makes the estimate noisy; its
across-seed standard deviation grows as the threshold shrinks. Direction 2
(target $p^2$) uses bounded weights $\propto p$, largest at the center where
close pairs are abundant, and is markedly more stable.

- **Direction 1 ($p$) is the scientifically natural target.** The Fano factor,
  noise correlation, and FEM fraction are properties of the neuron *under the
  actual viewing conditions*, whose eye-position distribution is $p$. A
  quantity reported over $p^2$ is "as if the animal fixated more tightly than
  it did," which would require a caveat on every number. We therefore
  headline Direction 1 for any reported values.
- **Direction 2 ($p^2$) is the stable cross-check.** Where Direction 1 is too
  noisy (eccentric cells, small thresholds, little data), Direction 2's
  bounded weights give a reliable, if differently-targeted, estimate.

The **gap** $\lvert(1-\alpha)_{\text{full}} - (1-\alpha)_{\text{central}}\rvert$
is itself informative: it is $\approx 0$ for a homogeneous (`flat`) cell and
grows with the spatial structure of the eye-sensitivity profile. Since the
two consistent targets coincide *iff* the stimulus is homogeneous, the gap
is a direct, model-free **measure of stimulus non-homogeneity**.

---

# 5. Consequences on synthetic and real data

## 5.1 The naive estimator fails; the matched estimator recovers truth (synthetic)

Pure-Poisson synthetic data (true noise correlation $0$, true Fano $1$) under
the §2 generator, with the three structured profiles, breaks the naive
estimator on all three reported quantities (Fig. 3):

- **$1-\alpha$** (panel A): the naive estimator **over-states** the FEM
  fraction for central cells and **under-states** it — to the point of an
  undefined, negative $C_{\text{rate}}$ (NaN) — for eccentric cells. The
  matched estimator (target $p$) lands on the truth.
- **Noise correlation** (panel B): the rate-variance distribution mismatch
  leaks into $C_{\text{total}}-C_{\text{rate}}$, producing **spurious**
  stimulus-independent correlations (median $\lvert r\rvert$ far from $0$)
  where the truth is exactly $0$.
- **Fano factor** (panel C): the same leak biases the Fano factor away from
  $1$, with a population that splits off toward inflated values.

The sign rule is set by where each cell's eye-sensitivity lives: conditioning
on $\Delta e<\varepsilon$ weights the FEM by $p(e)^2$, so a profile whose
sensitivity is **central** is *over*-weighted (since $p^2$ emphasizes the
center) and one whose sensitivity is **eccentric** is *under*-weighted.

![**Figure 3 — The naive (distribution-unmatched) estimator fails on all
three reported quantities**, with a sign set by each cell's eye-sensitivity
profile. Pure-Poisson synthetic (true noise correlation $0$, true Fano $1$);
"matched" is the $p$-target corrected estimator. **(A)** $1-\alpha$ vs ground
truth: naive (×) biased, matched (○) on the identity line. **(B)** spurious
noise correlation. **(C)** biased Fano factor.](figures/fig_naive_failure.png)

The matched estimator recovers each target's own closed-form decomposition
(`test_estimators.py::test_full_target_recovers_p_decomposition` and
`::test_central_target_recovers_p2_decomposition`), up to a small
finite-threshold smoothing that shrinks as the threshold shrinks. Figure 4
illustrates the recovery and the Direction-1/Direction-2 tradeoff.

![**Figure 4 — The matched estimator recovers ground truth and exposes the
Direction-1/Direction-2 tradeoff.** **(A)** `full` recovers the $p$
decomposition and `central` the $p^2$ decomposition (identity line). **(B)**
For an eccentric cell, Direction 1's unbounded $1/p$ weights make $1-\alpha$
noisy as the threshold shrinks; Direction 2 is stable. **(C)** The
full-vs-central gap is $\approx 0$ for a homogeneous cell and grows with
non-homogeneity — a model-free diagnostic.](figures/fig_correction.png)

## 5.2 Real-data consequences (`fixRSVP`, cache-only)

We applied the matched estimator to the real `fixRSVP` recordings, cache-only
(no GPU, no model inference; `generate_realdata.py` reads the Figure-4 cache
of trial-aligned spikes and real eye trajectories). $1-\alpha$ and the Fano
factor are computed on the real spikes with each cell's own validity mask
(reproducing the Figure-2 per-cell $1-\alpha$ at the median); the
eye-position density is a Gaussian KDE of the measured fixational positions.
Pooled over **397 good cells** ($\mathrm{cc}_{\max}>0.85$, 2 monkeys, 24
sessions):

| quantity | naive | Direction 1 ($p$, `full`) | Direction 2 ($p^2$, `central`) |
|---|---|---|---|
| median $1-\alpha$ | **0.734** | 0.702 | 0.608 |
| median Fano | 0.846 | 0.875 | — |

- **The naive bias on population $1-\alpha$ is small.** The naive median
  (0.734) reproduces the Figure-2 value (0.732) and lies only $+0.022$ above
  the Direction-1-corrected value (0.702). On the actual-viewing target $p$,
  the existing Figure-2 / Figure-4 panel-D $1-\alpha$ conclusions are
  therefore **robust** to the (A2) distribution mismatch at the population
  level — real V1 cells do not behave like the extreme synthetic
  `central`/`eccentric` profiles, which suffer large biases.
- **But there is measurable non-homogeneity.** The gap between the two
  consistent targets, $\lvert(1-\alpha)_{\text{full}} -
  (1-\alpha)_{\text{central}}\rvert$, has a population **median of 0.089**
  with a tail beyond $0.3$ (Fig. 5B). Since the two targets would coincide
  for a homogeneous stimulus, this gap is direct evidence that the `fixRSVP`
  stimulus is non-homogeneous for a substantial fraction of cells, and it
  sets the scale at which the *choice* of eye-position distribution matters
  for $1-\alpha$.
- **The Fano factor shifts modestly** under matching (median
  $0.846\to0.875$, $+3\%$), consistent with the synthetic prediction that
  the Fano factor inherits the rate-variance distribution mismatch
  (Fig. 3C); per-cell shifts are larger.

![**Figure 5 — The correction on real data (397 good cells, cache-only).**
**(A)** $1-\alpha$ on real spikes: Direction 1 (blue) tracks the naive
estimate closely (median shift $-0.022$), while Direction 2 (red) is
systematically lower. **(B)** The full-vs-central gap — a model-free
non-homogeneity measure — has median $0.089$ with a heavy tail. **(C)** Fano
factor: naive vs matched, a modest median shift with larger per-cell
changes.](figures/fig_realdata.png)

The **noise-correlation** consequence on real spikes is established on
synthetic ground truth (Fig. 3B), where the truth is exactly zero and the
naive estimator produces large spurious correlations that the matched
estimator removes. Quantifying it on real spike pairs requires either the
full windowed pipeline or careful joint-pair validity masking; because it
shares the same pipeline change, it is folded into the gated Figure-2 fix
(§6) rather than approximated here.

---

# 6. The current pipeline: state and proposed production change

The covariance machinery and caches are shared by Figures 2–4, so a pipeline
change has broad blast radius. This section consolidates the
production-pipeline state for each extension.

## 6.1 Extension 1 — already integrated

Consistent pair-count phase weighting has been integrated into the production
pipeline (`VisionCore/covariance.py`):

- `estimate_rate_covariance` — pair-count-weighted $\bar Y$ matching the
  close-pair second moment, fixing Mismatch 1 of §3.1.
- `bagged_split_half_psth_covariance` — `weighting` parameter, default
  `'pair_count'`, fixing Mismatch 2 of §3.1.

On the Allen 2022-04-13 session (49 cells, 3667 windows) the fix moved the
shuffle-null $D_z$ from $-0.0068$ ($p<10^{-4}$) to $+0.0010$ ($p=0.44$); the
real-data $D_z$ shifted only $-0.0855 \to -0.0819$ (signal-to-null ratio
$12.5\times \to 83.7\times$), so the scientific conclusion was preserved.
Diagnosis and validation are recorded in
`ryan/fig2/bias_diagnosis/FINAL_REPORT.md`.

## 6.2 Extension 2 — implemented here, production gated

Eye-position-distribution matching is implemented and TDD-validated in this
folder:

- `estimators.decompose(target=...)` — `naive` reproduces the existing
  pipeline; `full` is Direction 1 (the actual-viewing $p$); `central` is
  Direction 2 ($p^2$).
- `test_estimators.py` — 11 tests covering correctness (recovery of the
  closed-form decompositions under each target, finite-threshold-bias
  shrinkage), stability (Direction 2 stabler than Direction 1 for eccentric
  cells), Poisson cancellation (Fano $\to 1$), the pipeline-match
  (`naive` ↔ `pipeline_one_minus_alpha`), and Extension 1
  (variable-$n_t$ uniform vs pair-count weighting).

The proposed production change is to add the `target` argument to
`estimate_rate_covariance` (close-pair weight $q/p^2$) and a corresponding
per-sample weight $q/p$ to `bagged_split_half_psth_covariance` and the
$C_{\text{total}}$ computation in `VisionCore/covariance.py`, defaulted to
`'naive'` so the current numbers are preserved unless `target='full'` is
explicitly requested.

The regeneration of the Figure-2 decomposition cache requires a GPU and is
expensive; **it is gated on explicit approval** and is not performed by this
note. The expected direction of the change on $1-\alpha$ is set by §5.2:
small at the population median, larger for the heavy-tail non-homogeneous
cells.

---

# Appendix: derivations

## A.1 Close-pair density is $p(e)^2$

Let $e_i,e_j\stackrel{\text{iid}}{\sim}p$. The probability that a pair is
"close" with both members in a small ball $B_\delta(e)$ around $e$ is
$\Pr[e_i\in B_\delta(e)]\,\Pr[e_j\in B_\delta(e)] \approx \big(p(e)\,
|B_\delta|\big)^2$. Dividing by the total close-pair mass and taking
$\delta\to0$ gives the close-pair position density $p(e)^2/\int p^2$. For
$p=\mathcal N(0,\sigma^2 I)$ in $d$ dimensions, $p(e)^2 \propto
\exp(-\lVert e\rVert^2/\sigma^2)=\mathcal N(0,\tfrac{\sigma^2}{2}I)$ up to
normalization — variance halved.

## A.2 Importance weights

To estimate $\mathbb E_q[g] = \int g\,q$ from samples distributed as $s$,
write $\mathbb E_q[g] = \int g\,\tfrac{q}{s}\,s = \mathbb E_s\!
\big[g\,\tfrac{q}{s}\big]$, so each sample is weighted by $q/s$. For the
second moment the close pairs have $s=p^2$, giving weight $q/p^2$
(Direction 1: $p/p^2=1/p$; Direction 2: $p^2/p^2=1$). For the
mean/total/PSTH the trials have $s=p$, giving weight $q/p$ (Direction 1:
$1$; Direction 2: $p$). Substituting into the empirical LOTC terms yields a
decomposition in which all of $C_{\text{total}}$, $C_{\text{psth}}$ and
$C_{\text{rate}}$ are taken over the single distribution $q$, restoring
term-by-term consistency of (1).

## A.3 Pair-count weighting from the close-pair second moment

The close-pair second-moment estimator (7) averages products
$Y_c^iY_c^j$ over distinct same-phase pairs at $\Delta e<\varepsilon$.
With $n_t$ trials at phase $t$, the number of available pairs is
$n_t(n_t{-}1)/2$. Taking the uniform average over pairs is therefore taking
the across-phase average with weight $w_t\propto n_t(n_t{-}1)/2$. For the
LOTC (1) to hold term-by-term, the PSTH variance estimator (5) and the
mean $\bar Y$ must use the same $w_t$. Under constant $n_t$ this $w_t$
becomes a constant and the choice is invisible; under variable $n_t$ it is
the unique weighting that makes $C_{\text{total}}$, $C_{\text{psth}}$ and
$C_{\text{rate}}$ consistent.

## A.4 When the corrections are a no-op

- **Extension 1.** If $n_t$ is constant across phases, $w_t$ becomes a
  constant and the pair-count and uniform weightings coincide; the
  correction is the identity.
- **Extension 2.** The correction is the identity whenever (A2) holds:
  $\mathbb E_t[r^k(t,e)]$ independent of $e$ makes $\mathbb E_{e\sim D}\!
  \big[\mathbb E_t[r^k]\big]$ the same for every $D$, so the importance
  weights cancel. The `flat`-profile test
  (`test_homogeneous_stimulus_correction_is_noop`) satisfies (A2) by the
  stronger property that the rate itself is independent of $e$ on every
  trial.

---

*Reproduce all figures and tests (from this folder):*

```bash
uv run python fig_mechanism.py
uv run python fig_phase_weighting.py
uv run python fig_naive_failure.py
uv run python fig_correction.py
uv run python generate_realdata.py          # cache-only; --recompute to rebuild
uv run --with pytest pytest test_estimators.py -q
```

*Build this note to a self-contained, offline HTML (math via MathML, images
inlined):*

```bash
pandoc writeup.md -s --mathml --self-contained -o writeup.html
```
