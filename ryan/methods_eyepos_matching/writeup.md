---
title: "Extending McFarland's cross-trial decomposition to non-uniform trials and non-homogeneous stimuli"
subtitle: "A methodological note"
author: "fem-v1-fovea methods note"
date: "2026-05-29"
header-includes: |
  <style>
  .numbered-equation { position: relative; }
  .numbered-equation .eqno {
    position: absolute; right: 0.5em; top: 50%;
    transform: translateY(-50%);
  }
  </style>
---

# Summary

McFarland, Cumming & Butts (2016) introduced a cross-trial,
eye-position-conditioned decomposition that separates the stimulus-driven and
stimulus-independent components of V1 responses in the presence of fixational
eye movements (FEMs). It is now the standard estimator for the FEM fraction
$1-\alpha$, the corrected noise covariance $C_{\text{noise}}^{\text{corr}}$, and the
Fano factor under FEMs. Two assumptions sit inside it:

- **(A1) Uniform trial/time-bin structure** — every analysis time bin $t$ has the
  same number of trials $n_t$, so the across-time-bin weighting is the same for
  every term in the law-of-total-(co)variance (LOTC).
- **(A2) Statistically stationary stimulus** — the across-time-bin distribution
  of $r(t,e)$ at a fixed eye position is the same at every $e$
  (equivalently, $\mathbb{E}_t[r^k(t,e)]$ is independent of $e$ for the
  moments used). It then does not matter *which* eye-position distribution
  the rate moments are averaged over, and the close-pair restriction in (7)
  is free.

Both assumptions fail in the structured `fixRSVP` stimulus used here.
Fixation durations vary across trials, so the per-time-bin trial count $n_t$ drops
across time bins; and the windowed natural-image stimulus is far from
translation-invariant, so the rate $r(t,e)$ depends on absolute eye position.
This note develops, validates, and discusses a methodological extension of
McFarland et al. for each assumption violation:

1. **Consistent time-bin weighting** under variable $n_t$. The close-pair rate
   estimator is intrinsically pair-count weighted ($\propto n_t(n_t{-}1)/2$)
   across time bins, while McFarland's literal PSTH term is uniform ($1/T$) and
   the mean is trial-count weighted. Under variable $n_t$ this mix biases the
   **cross-cell** covariance — the corrected noise correlation is pushed
   negative and the shuffle-null control deviates from zero — while the per-cell
   $1-\alpha$ is barely affected. The bias is **additive, not multiplicative**
   in origin (a gain envelope is weighting-invariant; a random per-cell onset
   transient is not), and is removed by pinning every term to one weighting.
   The two consistent choices ($n_t$ and $1/T$) are equivalent in the model; we
   default to $n_t$ for its lower finite-sample variance. Demonstrated on
   synthetic ground truth and on 25 real `fixRSVP` sessions.
2. **Eye-position-distribution matching** under a non-homogeneous stimulus.
   Close pairs at threshold $\Delta e<\varepsilon$ are sampled in proportion
   to the *squared* eye-position density $p(e)^2$, while the total covariance
   and the PSTH covariance are over $p(e)$. We restore consistency with an
   importance-reweighted estimator that has the target eye distribution as a
   parameter, with two principled choices ($p$ and $p^2$). Their gap measures
   whether the rate has spatial structure on the fixation scale — non-zero
   even under (A2) when $\ell\sim\sigma_e$.

Sections 1 and 2 set up McFarland's estimator and the synthetic model that
breaks both assumptions. Sections 3 and 4 develop and validate one extension
each, on synthetic ground truth and on real `fixRSVP` data. A companion
implementation note (`note_pipeline.md`) records the production-pipeline
state — Extension 1 has already been integrated; Extension 2 is implemented
in this folder and gated.

---

# 1. Background: the cross-trial decomposition under FEMs

## 1.1 The law of total (co)variance

Let $Y_c^i(t)$ be the spike count of neuron $c$ on trial $i$ at analysis time bin
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
**distinct** trials, so for two different trials $i\neq j$ at the same time bin
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
over *all* distinct pairs at the same time bin, then over time bins, estimates the
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

Because *all* same-time-bin pairs are used, the per-bin mean averages over the
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

## 1.5 Three estimators on two weighting axes

Each estimator above averages distinct-trial products in a different way.
Reading directly off (5)–(8) and the LOTC LHS of (1), each lives on an
implicit weighting in two dimensions — across analysis time bins ($w_t$) and
across eye positions ($q$):

| Estimator | implicit $w_t$ (across-bin) | implicit $q$ (across-eye) |
|---|---|---|
| $C_\text{total}$ — LHS of (1); sample variance over all $(i,t)$ | $\propto n_t$ (trial-count) | $p$ |
| $C_\text{psth}$ — Eq. (6): $\langle\langle Y^i Y^j\rangle_{i\ne j}\rangle_t - \bar Y\bar Y^\top$ | $1/T$ for the $\langle\langle Y^iY^j\rangle\rangle$ term; $\propto n_t$ for $\bar Y\bar Y^\top$ | $p$ |
| $C_\text{rate}$ — Eq. (8): $\langle\langle Y^i Y^j \mid \Delta e<\varepsilon\rangle_{i\ne j}\rangle_t - \bar Y\bar Y^\top$ | $1/T$ for the $\langle\langle Y^iY^j\rangle\rangle$ term; $\propto n_t$ for $\bar Y\bar Y^\top$ | $p^2$ for the $\langle\langle Y^iY^j\rangle\rangle$ term; $p$ for $\bar Y\bar Y^\top$ |

The notation in the cells:

- **$w_t = 1/T$ ("uniform across bins").** Bin $t$ contributes $1/T$
  regardless of $n_t$ — compute a per-bin mean first, then average the per-bin
  means uniformly across $T$ bins. This is McFarland's literal reading of the
  nested bracket $\langle\langle\,\cdot\,\rangle_{i\ne j}\rangle_t$ in (6) and
  (8): the inner $\langle\cdot\rangle_{i\ne j}$ averages within bin $t$, the
  outer $\langle\cdot\rangle_t$ averages bins uniformly.
- **$w_t \propto n_t$ ("trial-count").** Bin $t$ contributes weight
  proportional to its trial count — pool all $(i, t)$ samples and weight each
  equally; bin $t$ then appears $n_t$ times in the pool and contributes $n_t$
  units of weight. This is how the sample variance of $Y$ pooled over $(i,t)$
  is computed (so $C_\text{total}$ lives here), and how the global mean
  $\bar Y = \frac{1}{\sum_t n_t}\sum_{i,t} Y^i(t)$ lives (so the
  $\bar Y\bar Y^\top$ subtractor lives here).

Concretely: under the fixRSVP staircase $n_t$ ranging $15 \to 360$, a
360-trial bin and a 15-trial bin contribute equally under $w_t = 1/T$, but
the 360-trial bin contributes $24\times$ more weight under
$w_t \propto n_t$. The two weightings coincide only when $n_t$ is
constant — then $1/T$ and $n_t / \sum n_t$ are the same number for every bin.

The across-eye axis $q$ has only two values in the table, and they live on
different cells for a concrete sampling reason:

- **$q = p$ ("full fixational density").** The across-eye distribution is the
  actual viewing distribution $p(e)$ each trial samples its eye position from.
  Any average that uses *all* same-time-bin trials averages the rate over $p$:
  the global mean $\bar Y$, $C_\text{total}$, and the all-pairs PSTH
  second moment (6) all live at $q = p$.
- **$q = p^2$ ("close-pair density").** The close-pair restriction
  $\Delta e_{ij} < \varepsilon$ in (7)–(8) keeps only pairs whose two eye
  positions nearly coincide. For $e_i, e_j$ drawn independently from $p$, the
  probability that both land in a small ball around $e$ is
  $\propto p(e)\cdot p(e) = p(e)^2$, so the close-pair second moment averages
  $r(t,e)^2$ over the **squared** density
  $p^2(e) \equiv p(e)^2 / \int p(e')^2\,de'$, not over $p$. For an isotropic
  Gaussian $p = \mathcal N(0,\sigma_e^2 I)$ this is *exactly*
  $\mathcal N(0,\tfrac{\sigma_e^2}{2} I)$ — a tighter, more central Gaussian with
  half the variance. This single fact (the homogeneity caveat McFarland's
  stationary stimulus lets him sidestep) is the entire origin of the $q$-axis
  mismatch, and §4 is built on it.

Throughout, a superscript on $\alpha$ names the across-eye distribution the
ratio (2) is evaluated over: $\alpha^p$ (and $1-\alpha^p$) is McFarland's
quantity under the actual viewing density $p$, while $\alpha^{p^2}$ is the same
ratio evaluated under the close-pair density $p^2$. The two differ whenever the
rate's second moment depends on absolute eye position (the (A2) failure of §4);
they coincide under McFarland's stationary-stimulus assumption.

The cross-trial **second-moment term** $\langle\langle Y^iY^j\rangle\rangle$ —
the part that averages distinct-trial products — is distinct from the
$\bar Y\bar Y^\top$ subtractor because the two are computed from the data
differently (close-pair products at fixed $t$ vs the global mean over all
$(i,t)$), so they live on different $(w_t, q)$ cells.

The LOTC (1) holds term-by-term **only when all three estimators land on a
single $(w_t, q)$**. The literal McFarland forms do not: even inside one
estimator the second-moment term and the $\bar Y\bar Y^\top$ subtractor
disagree on $w_t$ (second-moment term at $1/T$, subtractor at $\propto n_t$).
And $C_\text{rate}$'s second-moment term is the only cell at $q = p^2$ (the
close-pair conditional density, §4.1); every other cell is at $q = p$.

Two assumptions in McFarland's regime make these inconsistencies invisible:

- **(A1) Uniform trial/time-bin structure.** When $n_t$ is constant,
  $\propto n_t$ and $1/T$ coincide up to a global scaling, so the $w_t$
  column collapses to a single value across all three estimators (and across
  the second-moment / $\bar Y$ split within each).
- **(A2) Statistically stationary stimulus.** The across-time-bin
  distribution of $r(t,e)$ at a fixed eye position is the same at every $e$,
  i.e. $\mathbb{E}_t[r^k(t,e)]$ is independent of $e$ for the moments used.
  Then $\mathbb{E}_{e\sim D}\!\big[\mathbb E_t[r^k(t,e)]\big]$ is the same
  for every distribution $D$, so the $q$ column collapses: the close-pair
  restriction in (7)–(8) — which silently sets $D = p^2$ — gives the same
  answer as the actual viewing target $D = p$. McFarland et al. state the
  assumption at the level of the stimulus (their text around Eqs. M7–M10):

  > "by restricting analysis to trial pairs where $\Delta e_{ij}\approx 0$,
  > [the estimator] gives an estimate of $\mathbb{E}[r^2(e,t)]$ under the
  > conditional eye position distribution $p(e\mid\Delta e\approx 0)$ rather
  > than $p(e)$. However, for a stimulus that is statistically invariant to
  > spatial translations (such as used in our study), the expectation of
  > $r^2(e,t)$ with respect to any distribution over $e$ will be the same."

  Stimulus stationarity is sufficient: when each frame is a sample from a
  translation-invariant random field, the rate distribution at one eye
  position is, in distribution, the rate distribution at any other, so the
  across-time-bin moments at fixed $e$ do not depend on $e$. McFarland's
  ternary bar noise has this property; the structured `fixRSVP` images used
  here do not.

In McFarland's regime — (A1) ∩ (A2) — every cell of the table collapses to a
single $(w_t, q)$, the three estimators agree, and (1) reduces to the
textbook LOTC. Outside that regime each axis fails on its own and the LOTC
fails term-by-term. §3 develops two consistent choices on the $w_t$ axis
under (A1) failure; §4 develops two consistent choices on the $q$ axis
under (A2) failure. The two extensions are orthogonal — each fills one
column of the table — and §2 makes both failures concrete on the `fixRSVP`
stimulus.

---

# 2. Our setting and the synthetic model

Having laid out the two assumption failures theoretically, we now build
intuition for exactly how they manifest, using an analytical model with a
closed-form solution for the PSTH and rate variance. The model lets us turn
each assumption on or off as a switch and read off the consequences against
ground truth — first describing where the `fixRSVP` setting breaks each
assumption (§2.1, §2.2), then the generator itself (§2.3), and finally a
confirmation that McFarland's estimator is exact when both assumptions hold
(§2.4).

## 2.1 Violation of (A1): uneven fixation durations $\Rightarrow$ variable $n_t$

`fixRSVP` trials are organized around fixational fixation periods of variable
duration. Aligned to fixation onset and binned into analysis time bins, this
gives a monotonically decreasing per-time-bin trial count: every trial is
fixating in early bins, but only long-fixation trials contribute to late bins.
On a representative session (Allen 2022-04-13, 49 cells) $n_t$ ranged from 42
to 78 across time bins (CV $\approx 0.19$); on our broader set of cells the
spread is larger. Under this $n_t$ variation, the close-pair rate estimator
weights each time bin by $n_t(n_t{-}1)/2$ pairs while a uniform split-half PSTH
estimator weights each time bin by $1/T$ — these only coincide if $n_t$ is
constant.

Variable fixation duration is the dominant cause of differing $n_t$ here, but
it is not the only one: anything that masks out individual (trial, time-bin)
samples produces the same effect. Varying neuron stability (a unit drifts in or
out of isolation over a session), or any quality gate applied per sample — most
commonly excluding samples whose eye position falls outside predefined bounds —
also leaves a time-bin-dependent count of valid samples. Extension 1 is
therefore the general fix for an *arbitrarily masked* sample array, of which the
variable-duration staircase is one instance; the synthetic below uses the
staircase as the cleanest controllable case.

## 2.2 Violation of (A2): the windowed `fixRSVP` stimulus

`fixRSVP` presents a natural image at a fixed location on the screen, behind a
spatial **window** (aperture). The neuron's receptive field (RF) is fixed in
retinotopic coordinates, so in screen coordinates it moves with the eye. Two
mechanisms make the **across-time-bin distribution** of $r_c(t,e)$ at fixed eye
position depend on absolute eye position — directly violating (A2):

1. **Windowing — a hard non-stationarity.** Once the fixation offset is large
   enough to carry the RF off the windowed image, the RF samples only the
   uniform gray background and the rate collapses to the gray-screen baseline
   **regardless of analysis time bin $t$**. So at peripheral $e$ the across-time-bin
   distribution of $r$ is concentrated at baseline; at central $e$ it is
   strongly stimulus-modulated. The across-time-bin moments at fixed $e$ are
   therefore strongly $e$-dependent — *before any image content is
   considered*.
2. **Image structure.** Within the window, drift slides the structured image
   across the RF, so which feature drives the cell — and how strongly —
   depends on absolute eye position. The across-time-bin distribution of $r$ at
   each $e$ samples a different patch of a fixed image, not different draws
   from a translation-invariant ensemble; the per-$e$ moments differ
   accordingly.

## 2.3 A unified generative model that breaks both assumptions

We validate every estimator claim below against a single synthetic generator
(`synthetic.py`) chosen so that (A1) and (A2) are switches on top of one
architecture. With both switches off, the generator sits squarely in
McFarland's regime, with a closed-form $1-\alpha^p$ that depends on a single
ratio — the rate's spatial scale relative to the fixation scale — and covers
$(0,1)$ as that ratio is swept.

![**Unified generative model — components and resulting data.**
$r_c(t,e) = \mu_0 + M_c(e)\,\alpha(t)\,s_t(e)$.
**(A)** Eye distribution $p(e) = \mathcal N(0,\sigma_e^2 I)$, $\sigma_e=0.15^\circ$;
dashed rings at $1\sigma_e,\,2\sigma_e$.
**(B)** One draw of the stationary Gaussian random field $s_t(e)$ with kernel
$K(\delta)=\tau^2\exp(-\lVert\delta\rVert^2/(2\ell^2))$ at $\ell=0.5\sigma_e$
(intentionally short to make spatial structure visible).
**(C)** The three spatial masks $M_c(e)$ — the (A2) switch — tiled into one
panel: `flat`, `central`, `eccentric`.
**(D)** Envelope effect on a single trial of a `central` cell: identical eye
trajectory and field draws under $\alpha\equiv 1$ (dashed) and a decaying
$\alpha(t)$ (solid). Only $\alpha$ differs, so the late-time compression
toward $\mu_0$ is the envelope alone.
**(E)** Full rate $r$ over $(\text{trial},\text{time bin})$ for one `central`
cell with constant $n_t$ — the $(N_\text{tr},T)$ array is a full rectangle.
**(F)** Same with variable $n_t$ (staircase $15\to 2$); cells past trial end
are NaN, rendered white. The step line traces the $n_t$ boundary — this is
the (A1) regime.](figures/fig_model.png)

**Rate field.** For neuron $c$, analysis time bin $t$, and absolute eye position
$e\in\mathbb R^2$,

$$
r_c(t,e) \;=\; \mu_0 \;+\; M_c(e)\,\alpha(t)\,s_t(e),
\tag{9}
$$

with three separable ingredients:

- $s_t(\cdot)$ — a per-time-bin i.i.d. draw of a **stationary 2-D zero-mean
  Gaussian random field** with covariance
  $K(\delta) = \tau^2\exp\!\big(-\lVert\delta\rVert^2/(2\ell^2)\big)$. This
  is the rate map at time bin $t$. At any fixed $e$,
  $s_t(e)\sim\mathcal N(0,\tau^2)$ across time bins — the across-time-bin
  distribution is independent of $e$, so the *field component* is (A2) by
  construction.
- $\alpha(t)$ — a per-time-bin *multiplicative* amplitude (gain) envelope.
  Default $\alpha\equiv 1$. When the synthetic targets Extension 1 (§3) we set
  $\alpha(t)$ to decay across time bins, co-varying with $n_t$, so the two
  consistent $w_t$ directions differ in finite-sample variance (§3.4). The gain
  envelope does *not* by itself bias the cross-cell covariance — it is
  weighting-invariant (§3.3); that bias requires an *additive* onset transient,
  introduced in §3.3 as a separate illustrative component on top of $\alpha(t)$.
- $M_c(e)$ — the per-cell **spatial mask** in $[0,1]$, the (A2) switch.
  Physically, $M_c(e)$ is the fraction of the windowed stimulus the cell
  sees at eye position $e$: when the RF leaves the window, $M(e)\to 0$ and
  the rate collapses to baseline regardless of time bin, which is precisely
  §2.2's "hard windowing" mechanism. $M\equiv 1$ recovers a fully
  homogeneous stimulus; any non-constant $M$ breaks (A2) at the second
  moment: $\mathbb E_t[r^2(t,e)]=\mu_0^2 + M(e)^2\,\mathbb
  E_t[\alpha^2]\,\tau^2$, which depends on $e$ through $M(e)^2$ —
  the exact form McFarland's caveat is about (their text around Eqs.
  M7–M10). The first moment $\mathbb E_t[r(t,e)] = \mu_0$ stays
  $e$-independent.

Three mask shapes span the relevant cases:

| mask kind | $M_c(e)$ | physical role |
|---|---|---|
| `flat` | $1$ | homogeneous stimulus — (A2) holds |
| `central` | $\exp\!\big(-\lVert e\rVert^2/(2\sigma_M^2)\big)$ | windowed stimulus: cell sees the stimulus near fixation, only baseline in periphery |
| `eccentric` | $1 - \exp\!\big(-\lVert e\rVert^2/(2\sigma_M^2)\big)$ | the bounded complement — cell suppressed at fixation, sees the stimulus when eye drifts |

`central` is the workhorse stand-in for the windowed `fixRSVP` mechanism.
`flat` is the regime in which McFarland's estimator is unbiased; we use it in
§2.4 to sanity-check the estimator against a non-trivial analytical
$1-\alpha^p$. `eccentric` stress-tests the corrected estimator on a mask whose
variance lives where close pairs are rare or biased.

**Eye distribution.** Eyes are drawn i.i.d. $e\sim p=\mathcal N(0,\sigma_e^2 I)$
per (trial, time-bin), $\sigma_e=0.15^\circ$ (realistic fixational drift). For an
isotropic Gaussian $p$, the close-pair density $p(e)^2$ is *exactly*
$\mathcal N(0,\sigma_e^2/2\,I)$ — a tighter Gaussian with half the variance —
so the LOTC under $p$ or $p^2$ has a closed form (Appendix §A.1).

**Trial structure.** The (trial, time-bin) array has shape $(N_{\text{tr}}, T)$.
With $n_{\text{tr/bin}}=\text{None}$ every time bin has $N_{\text{tr}}$
trials, recovering McFarland's uniform-trial regime. To violate (A1) we set
$n_{\text{tr/bin}}=(n_t)_{t=1}^{T}$ — a monotonically decaying staircase
(default lo $15$, hi $\sim N_{\text{tr}}$, see Fig. 1A) that mimics the
fixation-duration distribution. Entries beyond $n_t$ in each time bin are
masked invalid.

**Observation noise.** Spikes are drawn
$Y_c \sim \mathrm{Poisson}(r_c(t,e))$ (optionally with a shared latent
giving a known stimulus-independent noise covariance). The rate is
marginally $\mathcal N(\mu_0, M(e)^2\tau^2)$; with $\mu_0=6,\,\tau\leq 1$,
$\Pr[r<0]\sim 10^{-9}$ and the clip to $r\geq 10^{-6}$ effectively never
triggers. An exponential link is the standard alternative but obscures the
closed form.

**Stimulus-frame caveat.** The synthetic treats each analysis time bin $t$
as an independent fresh draw of the field $s_t$, mirroring McFarland's
"stimulus phase" abstraction. In real `fixRSVP` this is an idealization:
the natural-image stimulus refreshes at $20$ Hz while the analysis
typically operates at $60$ or $120$ Hz, so several consecutive analysis
time bins fall within the *same* $50$ ms stimulus-frame interval and
share the same underlying $s_t$ — they are highly correlated, not i.i.d.
Across stimulus-frame boundaries the rate is genuinely refreshed. The
practical consequence (developed in §A.9) is that the effective $T$
controlling the across-bin SEM floor is closer to
$\#\text{fixations}\times\#\text{stim-frames per fixation}$ than the raw
bin count $\#\text{fixations}\times\#\text{bins per fixation}$. Within-
stimulus-frame reliability — how the cell's response varies across the
several analysis bins of one frame — is a future direction; the
analyses below treat this as out-of-scope and use the i.i.d.-bin
synthetic as a clean reference for the McFarland-style estimator.

**Closed-form ground truth.** With $M$ depending only on $e$ and $\alpha$
only on $t$, and the field zero-mean, the LOTC decomposition admits a closed
form under any distribution $D$ over $e$ and any time-bin weighting $w_t$
(derivation in Appendix §A.1):

$$
\mathrm{Var}_{\text{total}}^{D,w} = \mathbb E_w[\alpha^2]\,\tau^2\,\mathbb E_D[M^2],
\qquad
\mathrm{Var}_{\text{psth}}^{D,w} = \mathbb E_w[\alpha^2]\,I_{M,K,D},
\tag{10}
$$

with $I_{M,K,D}=\iint M(e_1) M(e_2)\,K(e_1{-}e_2)\,D(e_1)\,D(e_2)\,de_1\,de_2$.
The ratio $1-\alpha^{D,w} = 1 - I_{M,K,D}/(\tau^2\,\mathbb E_D[M^2])$ is
**invariant under time-bin weighting**: the envelope factor cancels. Both
$w_t$ directions of Extension 1 therefore target the same truth on the ratio;
they differ only in finite-sample efficiency (§3.4).

For the `flat` mask and Gaussian $D, K$, the integral closes analytically:

$$
1-\alpha^p = \frac{2\sigma_e^2}{\ell^2 + 2\sigma_e^2},
\qquad
1-\alpha^{p^2} = \frac{\sigma_e^2}{\ell^2 + \sigma_e^2},
\tag{11}
$$

a single-parameter family in $\ell/\sigma_e$ that covers $(0,1)$. The `central`
(Gaussian) and `eccentric` (one-minus-Gaussian) masks close in the same way
(Appendix §A.3, §A.4); all three closed forms were checked against Monte Carlo
(sampling noise $\lesssim 10^{-3}$).

These closed forms already expose the heart of Extension 2 (§4), before any
estimator is introduced: **the decomposition depends on which eye-position
distribution it is evaluated over.** Even for the homogeneous `flat` mask,
$1-\alpha^p$ and $1-\alpha^{p^2}$ differ at every finite $\ell$ (Fig. 0a A) —
the close-pair density $p^2$ is tighter than the viewing density $p$, and a rate
map with spatial structure on the fixation scale ($\ell\sim\sigma_e$) integrates
to a different variance under each. Their gap peaks at $\ell/\sigma_e\approx 1.18$
and vanishes only in the two degenerate limits ($\ell\to 0$, fully decorrelated
rates; $\ell\to\infty$, a spatially uniform field). For a non-homogeneous mask
the dependence is amplified, and its size is set jointly by the mask width
$\sigma_M$ and $\ell/\sigma_e$: Fig. 0a B sweeps $\ell/\sigma_e$ for the `central`
(Gaussian) mask at several $\sigma_M$, and the $p$-vs-$p^2$ spread is largest
(reaching $\approx 0.2$) when the mask width is comparable to the fixation scale
($\sigma_M\sim\sigma_e$), shrinking for masks much narrower or much wider than it.
*Which* distribution is the right one — and how to estimate the
decomposition consistently over it — is the subject of §4; here we only note
that the choice is not free once the rate has fixation-scale structure.

![**Figure 0a — The closed-form decomposition depends on the eye-position
distribution and the mask width.** All curves are closed-form (Appendix
§A.1–§A.4), not estimator output. **(A)** `flat` mask: $1-\alpha^p$,
$1-\alpha^{p^2}$, and their gap vs $\ell/\sigma_e$. The two distributions give a
different $1-\alpha$ at every finite $\ell$; the gap peaks at
$\ell/\sigma_e\approx 1.18$ (max $\approx 0.17$) and vanishes only as
$\ell\to 0$ or $\ell\to\infty$. **(B)** `central` (Gaussian) mask:
$1-\alpha^p$ (solid) and $1-\alpha^{p^2}$ (dashed) vs $\ell/\sigma_e$ for mask
widths $\sigma_M\in\{0.4,0.6,1.0\}\,\sigma_e$. The $p$-vs-$p^2$ spread is set
jointly by $\ell/\sigma_e$ and $\sigma_M$, peaking when the mask width is
comparable to the fixation scale ($\sigma_M\sim\sigma_e$). **(C)** `eccentric`
(one-minus-Gaussian) mask on the same axes: the bounded-complement case, whose
modulated variance lives in the periphery rather than at fixation.](figures/fig_distribution_truth.png)

## 2.4 McFarland's estimator recovers analytical $1-\alpha^p$ under (A1)+(A2)

With both assumptions satisfied — constant $n_t$ and the homogeneous `flat`
mask — McFarland's close-pair estimator should return the analytical
$1-\alpha^p$ of Eq. (11). The `flat` mask is a clean test target precisely
because $1-\alpha^p$ traces the full $(0,1)$ range as $\ell/\sigma_e$ is swept,
so agreement across the whole sweep is a non-trivial check rather than a single
point.

The estimator is unbiased here despite drawing its close pairs from the tighter
$p^2$ density, and the reason is the (A2) degeneracy: with a constant mask and a
stationary field, $\mathbb E_t[r^2(t,e)]$ does not depend on $e$, so the
close-pair second moment converges to the same value $\tau^2 + \mu_0^2$ no
matter where its midpoints land. The rate variance then recovers
$\mathrm{Var}_\text{total}^p = \tau^2$ and the PSTH variance recovers its $p$
value $\tau^2\ell^2/(\ell^2+2\sigma_e^2)$, so
$1-\alpha\to 2\sigma_e^2/(\ell^2+2\sigma_e^2) = 1-\alpha^p$ — even though the
close-pair density is intrinsically $p^2$. This is exactly the degeneracy that
breaks once the mask is non-constant and the second moment becomes
$e$-dependent; that is the subject of §4, and the $p$-vs-$p^2$ spread already
visible in Fig. 0a is what the naive estimator will get wrong there.

Figure 0b confirms the recovery empirically: across an $\ell/\sigma_e$ sweep the
estimator sits on the analytical $1-\alpha^p$ curve (panel A), and the estimate
is robust to the close-pair threshold over a useful range (panel B).

The estimator's *consistency* (how the seed-to-seed SEM depends on the
trials-per-time-bin $n$ **and** the number of time bins $T$) is more subtle than
$1/\sqrt{n}$: an across-time-bin noise floor at
$\sqrt{2\alpha^2/(T{-}1)}$ kicks in once within-bin sampling is
adequate, and at high SEM the $[0,1]$ clipping of $\alpha$ introduces a
mean bias. This is treated separately in Appendix §A.9.

![**Figure 0b — McFarland's estimator recovers the analytical $1-\alpha^p$
under (A1)+(A2).** **(A)** The empirical close-pair estimator (red,
mean$\pm$sd across 6 seeds) sits on the analytical $1-\alpha^p$ curve (blue)
across an $\ell/\sigma_e$ sweep covering $(0,1)$. **(B)** Threshold robustness
over a useful range ($\ell=\sigma_e$).](figures/fig_sanity_check.png)

---

# 3. Extension 1: consistent time-bin weighting (the $w_t$ column)

McFarland's literal estimators implicitly average over analysis time bins with
three *different* weightings (the $w_t$ column of the §1.5 table). Under
variable $n_t$ these diverge and the decomposition is no longer a covariance
under any single weighting. The visible consequence is not the per-cell
$1-\alpha$ — where the bias is small — but the **cross-cell covariance**: the
corrected noise correlation is biased downward and the shuffle-null control is
driven negative. The culprit is **additive, not multiplicative**, across-bin
structure, and the fix is to pin every term to one weighting.

## 3.1 McFarland's literal estimators mix three weightings

Reading the $w_t$ column of the §1.5 table off the literal forms:

- **$C_\text{rate}$** (8), pooled over close pairs, weights bin $t$ by its
  close-pair count $\propto n_t(n_t{-}1)/2$ — **pair-count**.
- **$C_\text{psth}$** (6), McFarland's nested bracket
  $\langle\langle\,\cdot\,\rangle_{i\ne j}\rangle_t$, averages within $t$ then
  uniformly across bins — **$w_t = 1/T$**.
- **$\bar Y\bar Y^\top$** (and $C_\text{total}$), the global mean over $(i,t)$,
  weights bin $t$ by $n_t$ — **trial-count**.

These coincide only when $n_t$ is constant. Under the `fixRSVP` staircase ($n_t$
ranging $\sim 15\to 360$ across bins; Fig. 1A) they diverge — $C_\text{rate}$
emphasizes early high-$n_t$ bins, $C_\text{psth}$ weights every bin equally,
$\bar Y$ sits between — and the LOTC (1) fails term-by-term.

## 3.2 The consequence is clearest in the shuffle null

The cleanest diagnostic for the weighting mismatch is a **shuffle null**.
Permuting eye trajectories across trials destroys the eye–spike coupling, so the
close-pair $C_\text{rate}^{\text{shuf}}$ has no real eye dependence and must
converge to $C_\text{psth}$ — provided both use the same $w_t$. The statistic
$D_z = f_z(C_\text{noise}^{\text{corr}}) - f_z(C_\text{noise}^{\text{uncorr}})$,
with $C_\text{noise}^{\text{corr}} = C_\text{total} - C_\text{rate}^{\text{shuf}}$
and $C_\text{noise}^{\text{uncorr}} = C_\text{total} - C_\text{psth}$, then has a
known truth of exactly zero — independent of any (A2) effect, since shuffling
removes the eye dependence Extension 2 acts on (§4.1). Any deviation is pure
weighting mismatch.

On real `fixRSVP` (Fig. 1D; 25 sessions, 100 shuffles each, cache-only via
`compute_weighting_data.py`) the mixed weighting gives a per-session mean
$D_z^{\text{shuf}} = -0.020$, while consistent pair-count weighting pins it to
$-0.001$. The mismatch falls where it does the most damage — the **cross-cell**
covariance: under mixing the off-diagonal $C_\text{rate}^{\text{shuf}}$ exceeds
$C_\text{psth}$, so the corrected noise covariance is pushed down and the
corrected noise correlations are biased negative (median $+0.008$ mixed vs
$+0.019$ consistent). The pre-fix production pipeline lived in exactly this mixed
state — close-pair second moment at pair-count, $C_\text{psth}$ at uniform $1/T$,
$\bar Y$ at trial-count — and the fix pins all three to pair-count
(`note_pipeline.md` §6.1).

![**Figure 1 — Mixing time-bin weightings biases the cross-cell covariance; the
bias is additive, not multiplicative.** Consistent (pair-count $n_t$) in green,
mixed (McFarland-literal) in red. **(A)** Under variable $n_t$ the close-pair
$C_\text{rate}$ (pair-count) and McFarland's $C_\text{psth}$ (uniform $1/T$)
weight time bins differently; they coincide only at constant $n_t$. **(C)**
Synthetic shuffle-null $D_z$ (truth $=0$, independent fields): a multiplicative
gain envelope produces no bias under mixing, while adding a random per-cell
additive onset transient drives the mixed estimator negative — the bias is
additive (§3.3). **(D)** Real `fixRSVP` shuffle-null $D_z$ (25 sessions): the
same fingerprint — mixed negative ($-0.020$), consistent $\approx 0$
($-0.001$).](figures/fig_weighting_bias.png)

## 3.3 Modeling: the bias is additive, not multiplicative

What across-bin structure produces the mismatch? Write the close-pair rate
covariance as $C_\text{rate} = \mathrm{MM} - \bar Y\bar Y^\top$. With the second
moment $\mathrm{MM}$ pair-count weighted but $\bar Y$ trial-count weighted, the
off-diagonal becomes $\mathrm{MM}_{mn} - \bar Y_m^{\text{trial}}\bar
Y_n^{\text{trial}}$ instead of the consistent $\mathrm{MM}_{mn} - \bar
Y_m^{\text{pair}}\bar Y_n^{\text{pair}}$, an error of $\bar Y^\text{pair}_m\bar
Y^\text{pair}_n - \bar Y^\text{trial}_m\bar Y^\text{trial}_n$ — nonzero exactly
when the pair- and trial-count means differ.

A multiplicative gain does not make them differ. In the unified model the field
is zero-mean, so a gain envelope $\alpha(t)$ leaves the mean rate flat at $\mu_0$
for every $t$: $\bar Y^\text{pair} = \bar Y^\text{trial} = \mu_0$, no inflation —
a gain change is **weighting-invariant**. An **additive onset transient** does.
If each cell carries a decaying component $g_c(t)$ on top of the field, the
pair-count mean — which emphasizes early high-$n_t$ bins (Fig. 1A) — exceeds the
trial-count mean: $\delta_c \equiv \bar Y^\text{pair}_c - \bar Y^\text{trial}_c >
0$ for every cell. The off-diagonal inflation is then $\approx \mu_0(\delta_m +
\delta_n) + \delta_m\delta_n$, **positive for every pair** — coherent across the
population even when the transient amplitudes are random and **independent**
across cells, because the large baseline $\mu_0$ multiplies each cell's shift. No
cross-cell co-tuning is required; the bias is a property of each cell's own
across-bin mean.

Figure 1C confirms this on a synthetic with a known truth of zero (independent
flat cells, so the true noise correlation *and* the cross-cell PSTH covariance
are both zero). With only the multiplicative gain envelope, the mixed and
consistent shuffle-null estimators both sit at zero. Adding a random per-cell
additive onset transient reproduces the real signature: the mixed estimator goes
negative, the consistent one stays at zero. The transient is one plausible,
biologically realistic candidate for the real-data bias (V1 onset transients are
ubiquitous), not a fixture of the canonical generator; it leaves each cell's
marginal $1-\alpha$ untouched and acts purely through the across-bin mean.

## 3.4 Two consistent directions, equivalent in the model; default to $n_t$

Restoring consistency means pinning every term — both second moments and the
$\bar Y$ subtractor — to a single $w_t$. Two principled choices exist:
**uniform** ($w_t = 1/T$, McFarland's literal nested-bracket reading) and
**pair-count** ($w_t \propto n_t(n_t{-}1)/2$, the close-pair estimator's
intrinsic weighting). For the multiplicative structure of the model the two are
**equivalent**: the envelope factor $\mathbb E_w[\alpha^2]$ cancels in the
$1-\alpha$ ratio (10), so both recover the same closed-form truth (verified
across the variable-$n_t$ tests). They differ only in efficiency — the close-pair
second moment has sampling variance $\propto 1/|P_t|$ with $|P_t|\approx
n_t(n_t{-}1)/2\cdot\Pr[\Delta e<\varepsilon]$, so the inverse-variance-optimal
across-bin combination *is* pair-count, while uniform gives noisy low-$n_t$ bins
equal voice. **We default to pair-count ($n_t$)** for all reported quantities.
(An additive transient, §3.3, would make even the truth weakly $w$-dependent — a
further reason a single weighting must be fixed and reported.)

---

# 4. Extension 2: pinning the $q$ column

Section 3 filled the $w_t$ column of the §1.5 table by pinning all three
estimators to a single across-bin weighting. This section fills the $q$
column by pinning all three to a single across-eye distribution. The two
extensions are independent — §3 addressed (A1) failure (variable $n_t$); §4
addresses (A2) failure (non-homogeneous stimulus) — and the same
natural-vs-stable tradeoff appears on this axis: McFarland's literal close-pair
second moment is pinned to the close-pair conditional density $p^2$, the actual viewing
distribution is $p$, and §4.2 develops both consistent directions, with their
bias/variance tradeoff in §4.3.

## 4.1 Failures of the naive estimator

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

For an isotropic Gaussian fixation $p=\mathcal N(0,\sigma_e^2 I)$ the close-pair
distribution is *exactly* $p^2=\mathcal N\!\big(0,\tfrac{\sigma_e^2}{2}I\big)$:
a tighter, more central Gaussian with **half the variance**.

Combining §1 with the close-pair density above, in the non-homogeneous case
the *naive* pipeline measures

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
cells) or driven *negative* (eccentric cells). Figure 2C shows both
directions on the closed-form decomposition; §4.2 confirms the empirical
recovery.

Note that this is a different axis from Extension 1: Extension 1's
shuffle null is **blind** to (A2) because shuffling eye trajectories
across trials destroys the eye–spike coupling, removing exactly the
$r(e)$ dependence on which the distribution mismatch acts. A
shuffle-null $D_z\approx 0$ therefore says nothing about eye-position
consistency.

Figure 2 collects the geometry and its consequence: the central
concentration of the close-pair density (A, B), and the resulting closed-form
$1-\alpha$ bias across mask kinds (C). The bias is negligible for the
homogeneous `flat` mask, where $\mathbb E_{p^2}[M^2]=\mathbb E_p[M^2]$ and the
decomposition is consistent on either distribution, and grows with the mask's
dependence on absolute eye position.

![**Figure 2 — Close pairs sample the squared density $p(e)^2$, and this
shifts $1-\alpha$ in a mask-dependent direction.** **(A)** Analytical eye
density $p(e) = \mathcal N(0, \sigma_e^2 I)$ with $\sigma_e = 0.15^\circ$
(grayscale), with iso-density contours at $1,2\sigma_e$ for $p$ (solid) and
for the close-pair density $p^2 = \mathcal N(0, \sigma_e^2/2\,I)$ (dashed) at
each distribution's own characteristic scale — the dashed contours are
tighter (variance halved). **(B)** The $x$-marginal: the close-pair
distribution matches $\mathcal N(0,\sigma_e^2/2)$ (observed variance ratio
$\approx 0.5$). **(C)** Closed-form $1-\alpha$ at $\ell=\sigma_e$,
$\sigma_M=\sigma_e$ for the unified rate-field model, across the three
masks. Blue: the truth $1-\alpha^p = 1 - I_{M,K,p} / (\tau^2\,\mathbb E_p[M^2])$.
Red: the naive estimator $1-\alpha^{\mathrm{naive}} = 1 - I_{M,K,p} / (\tau^2\,
\mathbb E_{p^2}[M^2])$, which keeps the correct PSTH numerator but uses the
close-pair $\mathbb E_{p^2}[M^2]$ in the denominator. For the `flat` mask
$\mathbb E_{p^2}[M^2]=\mathbb E_p[M^2]$ and the naive bias is negligible; the
bias is upward for `central` (denominator inflated) and downward for
`eccentric` (denominator deflated).](figures/fig_mechanism.png)

## 4.2 The corrected estimator: target distribution as a parameter

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

**What gets evaluated in practice.** The table above already exhibits the
key simplification: although the close-pair sampling density is $p(e)^2$,
the importance ratios for both consistent targets reduce to expressions in
$p$ alone, so the implementation never has to evaluate $p^2$ (nor its
unknown normalizer $\int p^2$) directly. We fit $\hat p(e)$ on the realized
per-(trial, time-bin) eye-position pool by Gaussian KDE
(`scipy.stats.gaussian_kde`, Scott's-rule bandwidth) — for synthetic and
real `fixRSVP` data alike — and evaluate it at close-pair midpoints for
Direction 1's $1/\hat p$ weights or at per-trial positions for Direction
2's $\propto\hat p$ weights. Both estimators are *self-normalized*
(`pw / pw.sum()` in the close-pair second moment;
`w / w.sum()` in the weighted mean and covariance helpers of
`estimators.py`), so the unknown $p^2$ normalizer and any uniform
mis-scaling of $\hat p$ cancel in numerator and denominator. *Shape* errors
in $\hat p$ do not cancel — they amplify in Direction 1 through the
$1/\hat p$ factor wherever close pairs are rare (the periphery), and are
the mechanical origin of the variance penalty discussed in §4.3.

Figure 3 confirms the construction on synthetic ground truth. Across an
$\ell/\sigma_e$ sweep for a `central` mask (left) and an `eccentric` mask
(right), Direction 1 (`full`) tracks the analytical $1-\alpha^p$ and
Direction 2 (`central`) tracks $1-\alpha^{p^2}$, while the naive estimator
departs from both in the mask-dependent direction of §4.1 — upward for
`central`, downward for `eccentric`.

![**Figure 3 — The matched estimators recover the analytical decomposition
under each target distribution.** Closed-form $1-\alpha^p$ (solid) and
$1-\alpha^{p^2}$ (dashed) vs $\ell/\sigma_e$ for the unified rate-field model
with a `central` mask (**A**) and an `eccentric` mask (**B**), at
$\sigma_M=\sigma_e$ ($N=800$ trials). Overlaid points are the empirical estimator at matched
$\ell/\sigma_e$ (mean $\pm$ sd across seeds): Direction 1 (`full`, $p$-weighted)
on the $1-\alpha^p$ curve, Direction 2 (`central`, $p^2$-weighted) on the
$1-\alpha^{p^2}$ curve, and the naive estimator biased off both — upward for
`central`, downward (clipping toward $0$) for `eccentric`.](figures/fig_recovery.png)

## 4.3 Direction 1 vs Direction 2: tradeoff and fixation-scale spatial-structure measure

The two consistent directions are not equally easy to estimate. Direction 1
(target $p$) requires the unbounded weight $1/p$, largest in the periphery —
exactly where close pairs are rarest. For an eccentric-modulated cell whose
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
is itself informative — but what it measures is *not* whether (A2) holds. The
unified random-field model under (A2) (`flat` mask) gives a closed-form gap

$$
(1{-}\alpha^p) - (1{-}\alpha^{p^2})
 = \frac{\sigma_e^2\,\ell^2}{(\ell^2 + 2\sigma_e^2)(\ell^2 + \sigma_e^2)},
\tag{16}
$$

which is **non-zero for finite $\ell$**, vanishing only as $\ell\to 0$
(decorrelated rates, all FEM under every $D$) or $\ell\to\infty$ (uniform
field, no FEM). The maximum is $\approx 0.17$ at $\ell/\sigma_e\approx 1.18$
(Fig. 0a A). The gap therefore measures whether the cell's **rate has spatial
structure on the fixation scale**: small gap when the rate is essentially
constant within a fixation ($\ell\gg\sigma_e$) or essentially decorrelated
within a fixation ($\ell\ll\sigma_e$); large gap when the rate's spatial
scale and the fixation scale are comparable. Non-homogeneous masks
(`central`, `eccentric`) **add** to this baseline.

The gap is still a useful empirical signal — it tells you that the *choice*
of eye-position distribution $D$ matters for the reported $1-\alpha$ — but
it is not a clean test of (A2).

## 4.4 Multi-bin eye trajectories: the production-setting extension

§4.2 framed the close-pair filter as a single-bin condition
$\lvert e_i-e_j\rvert<\varepsilon$ — and the §4.5 real-data analysis honours
that framing by using the eye position at one analysis time bin per sample.
The production covariance pipeline
(`VisionCore/covariance.py::compute_eye_distances`) instead works on
**$T$-bin trajectories**: each sample $i$ carries an eye trajectory
$\tau_i = (e_{i,1},\ldots,e_{i,T})$ over a window of $T = t_{\mathrm{hist}}+t_{\mathrm{count}}$
bins, and the close-pair filter is the *root-mean-square trajectory distance*

$$
\big\lVert\tau_i-\tau_j\big\rVert_{\mathrm{RMS}}
 \;=\; \sqrt{\tfrac{1}{T}\sum_{t=1}^{T}\lVert e_{i,t}-e_{j,t}\rVert^2}
 \;<\;\varepsilon
\tag{17}
$$

so that two trials are "close" only when their *whole* trajectories — not just
the spike-count bin — are similar. The motivation is that the neuron integrates
over a temporal window, so the response context that a close pair has to share
is the whole window, not the instantaneous eye position.

This breaks the §4.2 importance-weight construction as written. The close-pair
density now lives on $\mathbb R^{2T}$ — the trajectory density $p(\tau)$
squared, restricted to $\lVert\tau_i-\tau_j\rVert_{\mathrm{RMS}}<\varepsilon$ — and
fitting a density in $\mathbb R^{2T}$ for typical $T\in[10,30]$ is the curse of
dimensionality. We *cannot* simply lift the §4.5 single-bin KDE to the
trajectory and call it done.

**Our extension.** Build the importance weight from two 2-D KDEs:

$$
\hat p_{\mathrm{marg}}(e)\,=\,\mathrm{KDE}\big(\{e_{i,t}\}_{i=1\dots N,\,t=1\dots T}\big),
\qquad
\hat p_{cp,\mathrm{marg}}(e)\,=\,\mathrm{KDE}\big(\{m_{k,t}\}_{k=1\dots P,\,t=1\dots T}\big)
\tag{18}
$$

where $\hat p_{\mathrm{marg}}$ pools per-bin positions across *all* samples and
$\hat p_{cp,\mathrm{marg}}$ pools per-bin positions of the *close-pair midpoint
trajectories* $m_k=\tfrac12(\tau_i+\tau_j)$ (one midpoint trajectory per close
pair $k$, contributing $T$ per-bin positions to the pool). The §4.2 importance
weights are then evaluated at each trajectory's **centroid**
$c_i=\tfrac1T\sum_t e_{i,t}$:

| target | per-sample weight at $c_i$ | per-pair weight at $c_{\mathrm{mid}}=\tfrac12(c_i+c_j)$ |
|---|---|---|
| **Direction 1** (`full`) | $1$ | $\hat p_{\mathrm{marg}}(c_{\mathrm{mid}})\,/\,\hat p_{cp,\mathrm{marg}}(c_{\mathrm{mid}})$ |
| **Direction 2** (`central`) | $\hat p_{cp,\mathrm{marg}}(c_i)\,/\,\hat p_{\mathrm{marg}}(c_i)$ | $1$ |
| naive | $1$ | $1$ |

(`estimators.decompose_trajectory`).

**Why this works in the flat-trajectory limit.** Decompose
$e_{i,t}=c_i+\xi_{i,t}$ with $\xi_{i,t}\sim\mathcal N(0,\sigma_{\mathrm{drift}}^2 I)$
i.i.d. across $t$ and centroid $c_i\sim p_{\mathrm{centroid}}$. As
$\sigma_{\mathrm{drift}}\to 0$ the trajectory collapses to its centroid, so

$$
\hat p_{\mathrm{marg}}(e) \;\to\; p_{\mathrm{centroid}}(e),
\qquad
\hat p_{cp,\mathrm{marg}}(e) \;\propto\; p_{\mathrm{centroid}}(e)^2
\tag{19}
$$

(the latter because in the flat limit $p_{cp}(\tau)\propto p(\tau)^2$ is
supported on constant trajectories with weight $p_{\mathrm{centroid}}(c)^2$, and
pooling per-bin positions across midpoint trajectories collapses to a single
$p_{\mathrm{centroid}}^2$ KDE). The ratio $\hat p_{cp,\mathrm{marg}}/\hat p_{\mathrm{marg}}$
at the centroid then collapses to $p_{\mathrm{centroid}}(c)$ up to a normalising
constant, and the table above recovers §4.2's Direction 1 pair weight
$1/p_{\mathrm{centroid}}(c)$ and Direction 2 sample weight $\propto p_{\mathrm{centroid}}(c)$
exactly. Self-normalisation absorbs the normalising constant in numerator and
denominator.

**The flat-trajectory approximation.** For $\sigma_{\mathrm{drift}}>0$ the
estimator targets the *per-bin marginal* $p_{\mathrm{pb}}(e)=p_{\mathrm{centroid}}*\phi_{\sigma_{\mathrm{drift}}}(e)$
— a Gaussian-smoothed version of the centroid distribution. The "actual viewing
distribution" the estimator aims at is therefore $\mathcal N(0,\sigma_{\mathrm{traj}}^2 I)$
with $\sigma_{\mathrm{traj}}^2=\sigma_e^2+\sigma_{\mathrm{drift}}^2$, and the truth to
compare against is §4.2's closed form at $\sigma_{\mathrm{traj}}$
(`synthetic.ground_truth(kind, sqrt(sigma_e^2+sigma_drift^2), ell, sigma_M=sigma_M)`). The
construction is *exact* in expectation in the flat limit; for non-zero drift
the residual comes from two sources:

1. **Centroid-vs-per-bin smoothing.** Evaluating the ratio at the trajectory
   centroid (a single 2-D point) discards within-window drift; the ratio at
   $c_i$ is biased by an amount that scales with $\sigma_{\mathrm{drift}}/\sigma_e$.
2. **Threshold inflation.** Because $E[\lVert\tau_i-\tau_j\rVert^2_{\mathrm{RMS}}]
   \approx d_{\mathrm{centroid}}^2 + 4\sigma_{\mathrm{drift}}^2$, the RMS-trajectory
   threshold $\varepsilon$ must grow with $\sigma_{\mathrm{drift}}$ to admit any
   close pairs at all; at large $\sigma_{\mathrm{drift}}$ the close-pair filter
   loses selectivity and the central-region concentration of the close-pair
   midpoint density weakens.

**Validation (Fig. 4).** A controlled synthetic with explicit
$\sigma_{\mathrm{drift}}$ knob (`synthetic.make_trajectory_session`) shows the
flat-limit recovery is sharp and the bias grows smoothly with
$\sigma_{\mathrm{drift}}/\sigma_e$. At $\sigma_{\mathrm{drift}}/\sigma_e=0$ the corrected
Directions 1 and 2 sit on their respective truths within seed noise (panel
E); at $\sigma_{\mathrm{drift}}/\sigma_e\sim 0.2$ — comparable to the operating
regime for fixational drift over a typical $t_{\mathrm{hist}}+t_{\mathrm{count}}$
window — the bias is small; at $\sigma_{\mathrm{drift}}/\sigma_e\sim 1$ the
trajectories are no longer "essentially flat" and the bias is visible but
bounded. Panels B–D show the §4.1 mechanism reappearing in the multi-bin
setting: $\hat p_{cp,\mathrm{marg}}$ is narrower than $\hat p_{\mathrm{marg}}$, and
their ratio peaks at the centre — the close pairs over-represent the
high-density region exactly as the single-bin §4.1 picture predicts.

![**Figure 4 — The trajectory-mode estimator and its validation.** **(A)**
Example trajectories ($\sigma_{\mathrm{drift}}=\sigma_e/4$) showing centroid scatter
plus per-bin fixational drift. **(B)** $\hat p_{\mathrm{marg}}(e)$, the 2-D KDE
fit on pooled per-bin positions of all samples. **(C)** $\hat p_{cp,\mathrm{marg}}(e)$,
the 2-D KDE fit on per-bin positions of close-pair midpoint trajectories —
narrower than (B), concentrated at the centre. **(D)** The ratio
$\hat p_{cp,\mathrm{marg}}/\hat p_{\mathrm{marg}}$ — the §4.1 distribution mismatch in
the multi-bin setting, peaking at the centre as expected. **(E)**
$\sigma_{\mathrm{drift}}$ sweep on a `flat` mask: matched Directions 1 and 2 sit on
their respective truths (dotted lines) in the flat limit; bias grows smoothly
with $\sigma_{\mathrm{drift}}/\sigma_e$; naive over-states.](figures/fig_trajectory.png)

The trajectory-mode estimator is the production-setting bridge for §4.5's
single-bin analysis: when the §6.2 production change lands (`note_pipeline.md`),
the same
`target ∈ {'naive','full','central'}` parameter selects the same three
behaviours; the §4.5 numbers are recovered as the
$\sigma_{\mathrm{drift}}\to 0$, $T=1$ limit; and the curse-of-dimensionality wall
that the multi-bin filter raised is sidestepped by replacing the trajectory
density with two 2-D KDEs evaluated at the trajectory centroid.

### A note on the centred close-pair second moment

The implementation in `decompose_trajectory` computes the close-pair second
moment on *centred* counts $S - \hat r$ instead of the usual uncentred
$S_i S_j^\top$ followed by an $\hat r\hat r^\top$ subtraction. The motivation
is numerical: when $\hat r \cdot t_\text{window}$ dominates the magnitude of
$S$, the literal "second moment minus $\hat r\hat r^\top$" can be a
catastrophic cancellation, whereas centring first turns each pair product
into a small-times-small operation. This matters on the §4.4 synthetic
validation, where $\hat r$ is a known constant and the precision argument is
clean.

The two forms are related by the identity

$$\sum_{(i,j) \text{ close}} w_{ij}(S_i - \hat r)(S_j - \hat r)^\top
  = \widehat{MM} - \hat r\,\bar Y_w^\top - \bar X_w\,\hat r^\top + \hat r\hat r^\top,$$

where $\widehat{MM} = \sum w_{ij} S_i S_j^\top / \sum w_{ij}$ is the
weighted close-pair second moment and $\bar X_w = \sum w_{ij} S_i / \sum w_{ij}$,
$\bar Y_w = \sum w_{ij} S_j / \sum w_{ij}$ are the weighted close-pair-set
sample means of the left and right cell. They collapse to the uncentred form
$\widehat{MM} - \hat r\hat r^\top$ if and only if $\bar X_w = \bar Y_w = \hat r$ —
that is, if the mean we subtract equals the close-pair-set sample mean.

For the consistent targets (`full`, `central`) the importance reweighting
makes $\bar X_w$ and $\hat r$ both estimate the same population quantity
($\mathbb{E}_p[r]$ for full, $\mathbb{E}_{p^2}[r]$ for central) — but they
remain **distinct estimators in finite samples**: $\hat r$ averages all
samples under the per-sample weighting (low variance), while $\bar X_w$
averages the close-pair subset under the per-pair weighting (noisier — the
close-pair pool is sparse and the per-pair weights for `full` are a
density-ratio that the KDE only estimates). The centred and uncentred forms
therefore agree asymptotically (both consistent for $C_\text{rate}^q$) but
differ in finite samples by an amount controlled by
$\|\hat r - \bar X_w\|\,\|\hat r\|$. On the `note_pipeline.md` §7 real data this gap is
non-negligible — e.g. at $t_\text{count}=2$ the median $1-\alpha_\text{full}$
shifts from $0.71$ (centred) to $0.77$ (uncentred), and similarly for
central.

For the inconsistent target `naive` the gap is also non-zero, and for a
different reason: $\hat r$ estimates $\mathbb{E}_p[r]$ while $\bar X_w$
estimates $\mathbb{E}_{p^2}[r]$, so the two are not even equal in
expectation. The centred and uncentred naive estimators are therefore two
different inconsistent estimators of the §4.1 mixed quantity, with neither
privileged.

The `note_pipeline.md` §7 pipeline uses the **uncentred form for all three
targets** for two reasons:

  1. **§4.5 reference compatibility.** The cell-side single-bin analysis in
     §4.5 (`generate_realdata.py`, Fig. 5) ran on the single-bin
     `estimators.decompose`, which is uncentred. The `note_pipeline.md` §7
     multi-cell, multi-window pipeline numbers must extend §4.5 rather than
     redefine it, and this requires the same close-pair Crate semantic.
  2. **`note_pipeline.md` §7.2 equivalence audit.** Legacy `compute_conditional_second_moments`
     is uncentred, so `target='naive'` exact equivalence requires the
     uncentred form.

The centred form remains the default in
`estimators.decompose_trajectory` because the §4.4 synthetic validation
relies on its precision claim. Future work that promotes the trajectory-
mode estimator to production should decide centred vs uncentred per
target on its own finite-sample merits — the two are not interchangeable
on real data even where both are consistent.

## 4.5 Consequences on real data (`fixRSVP`, cache-only)

We applied the matched estimator to the real `fixRSVP` recordings, cache-only
(no GPU, no model inference; `generate_realdata.py` reads the Figure-4 cache
of trial-aligned spikes and real eye trajectories). $1-\alpha$ and the Fano
factor are computed on the real spikes with each cell's own validity mask
(reproducing the Figure-2 per-cell $1-\alpha$ at the median); the
eye-position density is a Gaussian KDE of the measured fixational positions.
This implementation uses a *single-bin* close-pair filter (the eye position
at one analysis time bin per sample), so the §4.2 importance weights apply
without modification; the multi-bin trajectory extension required by the
production filter (§4.4) is folded into the gated §6.2 pipeline change
(`note_pipeline.md`).
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
- **The gap measures fixation-scale spatial structure, and it is
  measurable.** The gap between the two consistent targets,
  $\lvert(1-\alpha)_{\text{full}} - (1-\alpha)_{\text{central}}\rvert$, has
  a population **median of 0.089** with a tail beyond $0.3$ (Fig. 5B).
  Under the unified random-field model the gap is non-zero under (A2)
  itself when $\ell\sim\sigma_e$ (Eq. 16) — peaking at $\approx 0.17$ — so
  gap $= 0.089$ is evidence that the cells' rate maps have spatial
  structure on the fixation scale, which is *expected* for any cell with a
  finite spatial RF, not necessarily that (A2) is violated. What the gap
  *does* say is that the **choice** of eye-position distribution $D$
  matters for the reported $1-\alpha$ at this scale: a $\pm 0.09$ swing
  between Direction 1 and Direction 2 is the order of the
  Direction-1-vs-naive bias we found above ($-0.022$).
- **The Fano factor shifts modestly** under matching (median
  $0.846\to0.875$, $+3\%$), consistent with the synthetic prediction that
  the Fano factor inherits the rate-variance distribution mismatch;
  per-cell shifts are larger.

![**Figure 5 — The correction on real data (397 good cells, cache-only).**
**(A)** $1-\alpha$ on real spikes: Direction 1 (blue) tracks the naive
estimate closely (median shift $-0.022$), while Direction 2 (red) is
systematically lower. **(B)** The full-vs-central gap — a measure of the
rate's spatial structure on the fixation scale — has median $0.089$ with a
heavy tail. **(C)** Fano
factor: naive vs matched, a modest median shift with larger per-cell
changes.](figures/fig_realdata.png)

The **noise-correlation** consequence on real spikes is small at the
population level — both on the unified synthetic (naive vs matched median
$|r|$ differ by $\approx 0.005$) and consistent with the modest Fano shift
here ($0.846\to 0.875$). Quantifying it directly on real spike pairs requires
either the full windowed pipeline or careful joint-pair validity masking;
because it shares the same pipeline change, it is folded into the gated
Figure-2 fix (`note_pipeline.md` §6) rather than approximated here.

---

# Appendix: derivations

The closed-form decomposition and its three mask evaluations come first
(§A.1–§A.4), followed by the sampling and weighting derivations the main text
relies on (§A.5–§A.8), the finite-sample consistency analysis (§A.9), and the
PSTH covariance estimator (§A.10).

## A.1 Closed-form decomposition for the unified rate field

With the unified rate equation (9), the across-time-bin and across-eye
distributions decouple by linearity. Write $X_t(e) = M(e)\,\alpha(t)\,s_t(e)$,
so $r = \mu_0 + X$. The field is zero-mean, $\mathbb E[s_t(e)] = 0$, with
covariance $K(\delta) = \tau^2\exp(-\lVert\delta\rVert^2/(2\ell^2))$ and
independent draws across time bins.

**Mean.** $\mathbb E_{t,e\sim D}[r] = \mu_0$ for any $w, D$: the field is
zero-mean and the mean is constant in $e$.

**Total variance.** $\mathrm{Var}_\text{total}^{D,w} = \mathbb E[X^2]
= \mathbb E_w[\alpha^2]\,\mathbb E_D[M^2]\,\mathbb E_s[s_t(e)^2]
= \mathbb E_w[\alpha^2]\,\tau^2\,\mathbb E_D[M^2]$.

**PSTH variance.** Let $G_t^D = \mathbb E_{e\sim D}[M(e)\,s_t(e)]$, so the
per-bin PSTH is $\mathbb E_{e\sim D}[r_t(e)] = \mu_0 + \alpha(t)\,G_t^D$. The
$G_t^D$ are i.i.d. across time bins with $\mathbb E[G_t^D] = 0$ and
$\mathbb E[(G_t^D)^2] = I_{M,K,D}$, so in the large-$T$ limit

$$
\mathrm{Var}_\text{PSTH}^{D,w} = \mathbb E_w[\alpha^2]\,I_{M,K,D},
\qquad
I_{M,K,D} = \iint M(e_1)\,M(e_2)\,K(e_1{-}e_2)\,D(e_1)\,D(e_2)\,de_1\,de_2.
$$

The ratio $1-\alpha^{D,w} = 1 - I_{M,K,D}/(\tau^2\,\mathbb E_D[M^2])$ is
**invariant under time-bin weighting** and the envelope $\alpha(t)$: both
$\mathrm{Var}_\text{PSTH}$ and $\mathrm{Var}_\text{total}$ carry the same factor
$\mathbb E_w[\alpha^2]$, which cancels. Both Extension-1 directions target the
same closed-form ratio; the choice between them is a finite-sample efficiency
question, not a bias question.

**A Gaussian two-point integral.** Every mask below is built from Gaussian
factors, so $I_{M,K,D}$ reduces to products of a single one-dimensional
two-point Gaussian integral. Take $D = \mathcal N(0, s^2 I)$ in $\mathbb R^2$;
the integrand factorises across the two spatial dimensions. The per-dimension
building block, carrying an extra Gaussian precision $c_1, c_2$ on its two
points, is

$$
T(c_1, c_2) = \iint e^{-\frac12 c_1 x_1^2}\,e^{-\frac12 c_2 x_2^2}\,
 e^{-(x_1-x_2)^2/(2\ell^2)}\,\mathcal N(x_1;0,s^2)\,\mathcal N(x_2;0,s^2)\,
 dx_1\,dx_2 = \frac{1}{s^2\sqrt{\det\Lambda}},
$$

$$
\Lambda = \begin{pmatrix} p_1 + \ell^{-2} & -\ell^{-2} \\
 -\ell^{-2} & p_2 + \ell^{-2}\end{pmatrix},
\quad p_k = \frac1{s^2} + c_k,
\quad \det\Lambda = p_1 p_2 + \frac{p_1 + p_2}{\ell^2}.
$$

The result follows by collecting the exponent into the quadratic form
$\tfrac12 x^\top\Lambda x$ and using
$\int e^{-\frac12 x^\top\Lambda x}\,dx_1\,dx_2 = 2\pi/\sqrt{\det\Lambda}$
against the $1/(2\pi s^2)$ from the two $\mathcal N(\cdot;0,s^2)$ normalisers. A
Gaussian mask factor $\exp(-\lVert e\rVert^2/(2\sigma_M^2))$ on a point sets
$c = 1/\sigma_M^2$ there; its absence sets $c = 0$. When the two-point product
$M(e_1)M(e_2)$ expands into a sum of such Gaussian-factored terms, the full
two-point integral is

$$
I_{M,K,D} = \tau^2 \sum_{\text{terms}} (\pm)\,\big[T(c_1, c_2)\big]^2,
$$

one squared $T$ per term (the square is the product over the two spatial
dimensions). The viewing and close-pair distributions enter only through $s^2$:
$s^2 = \sigma_e^2$ for $D = p$ and $s^2 = \sigma_e^2/2$ for $D = p^2$.

## A.2 Flat mask

For $M \equiv 1$ both points carry $c = 0$, so $I_{M,K,D}/\tau^2 = T(0,0)^2 =
\ell^2/(\ell^2 + 2s^2)$ and $\mathbb E_D[M^2] = 1$. With $s^2 = \sigma_e^2$ and
$s^2 = \sigma_e^2/2$ this gives

$$
1-\alpha^p = \frac{2\sigma_e^2}{\ell^2 + 2\sigma_e^2},
\qquad
1-\alpha^{p^2} = \frac{\sigma_e^2}{\ell^2 + \sigma_e^2},
$$

and the (A2)-respecting gap

$$
(1{-}\alpha^p) - (1{-}\alpha^{p^2})
 = \frac{\sigma_e^2\,\ell^2}{(\ell^2 + 2\sigma_e^2)(\ell^2 + \sigma_e^2)},
$$

whose maximum sits at $\ell/\sigma_e = 2^{1/4} \approx 1.19$ (differentiate in
$\ell^2$ and set to zero).

## A.3 Central mask

For $M(e) = \exp(-\lVert e\rVert^2/(2\sigma_M^2))$ both points carry the mask,
$c_1 = c_2 = 1/\sigma_M^2$, so $I_{M,K,D}/\tau^2 = T(1/\sigma_M^2,1/\sigma_M^2)^2$
and $\mathbb E_D[M^2] = \sigma_M^2/(\sigma_M^2 + 2s^2)$. Collecting the algebra
into the effective convolution scale
$\sigma_{\mathrm{eff}}^2 = s^2\sigma_M^2/(s^2 + \sigma_M^2)$,

$$
I_{M,K,D} = \tau^2 \left(\frac{\sigma_M^2}{s^2 + \sigma_M^2}\right)^{2}
 \frac{\ell^2}{\ell^2 + 2\sigma_{\mathrm{eff}}^2},
\qquad
1-\alpha^{D} = 1 - \frac{I_{M,K,D}}{\tau^2\,\mathbb E_D[M^2]},
$$

with $s^2 = \sigma_e^2$ for $D = p$ and $\sigma_e^2/2$ for $D = p^2$
(`synthetic._central_one_minus_alpha_closed_form`).

## A.4 Eccentric mask

For $M(e) = 1 - g(e)$ with $g(e) = \exp(-\lVert e\rVert^2/(2\sigma_M^2))$, the
two-point product expands as $M(e_1)M(e_2) = 1 - g_1 - g_2 + g_1 g_2$ and both
moments split into Gaussian pieces. The second moment is

$$
\mathbb E_D[M^2] = 1 - 2\,\mathbb E_D[g] + \mathbb E_D[g^2],
\qquad
\mathbb E_D[g] = \frac{\sigma_M^2}{\sigma_M^2 + s^2},
\quad
\mathbb E_D[g^2] = \frac{\sigma_M^2}{\sigma_M^2 + 2s^2}.
$$

Each term of the expansion carries the mask on neither, one, or both points,
mapping onto $T(0,0)$, $T(c,0)$ and $T(c,c)$ with $c = 1/\sigma_M^2$:

$$
\frac{I_{M,K,D}}{\tau^2} = T(0,0)^2 - 2\,T(c,0)^2 + T(c,c)^2,
\qquad
1-\alpha^{D} = 1 - \frac{I_{M,K,D}}{\tau^2\,\mathbb E_D[M^2]}
$$

(the two cross terms $-g_1$ and $-g_2$ are equal by symmetry, hence the factor
of two). This is `synthetic._eccentric_one_minus_alpha_closed_form`, and
`test_eccentric_mask_closed_form_matches_mc` checks it against a
$2\times10^6$-sample Monte-Carlo evaluation of the same ratio to
$<3\times10^{-3}$ across mask widths and both distributions.

## A.5 Close-pair density is $p(e)^2$

This short derivation has been lifted into the main text (§1.5, the
"$q = p^2$" bullet): for $e_i,e_j\stackrel{\text{iid}}{\sim}p$ the probability
that a pair is "close" with both members in a small ball $B_\delta(e)$ is
$\approx\big(p(e)\,|B_\delta|\big)^2$, so the close-pair position density is
$p(e)^2/\int p^2$, which for $p=\mathcal N(0,\sigma_e^2 I)$ is
$\mathcal N(0,\tfrac{\sigma_e^2}{2}I)$ — variance halved.

## A.6 Importance weights

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

## A.7 Two readings of the nested-bracket close-pair estimator

The close-pair second-moment estimator (7) is written
$\langle\langle Y_c^iY_c^j \mid \Delta e<\varepsilon\rangle_{i\ne j}\rangle_t$.
Two readings of the nested bracket give two consistent across-bin weightings.

**Inner-then-outer (uniform $1/T$).** Read the brackets literally as
two stages: at each bin $t$, average the close-pair products to get a
per-bin mean $\bar m_t = |P_t|^{-1}\sum_{(i,j)\in P_t} Y^iY^j$; then average
the $\{\bar m_t\}$ uniformly across the $T$ bins. Each bin contributes
$w_t = 1/T$ regardless of $|P_t|$. This is the natural reading of
"averaging across time points" — McFarland's text on p.6228.

**Pool-then-average (pair-count $\propto n_t(n_t{-}1)/2$).** Pool all close
pairs from all bins into a single flat list and average uniformly per pair.
Bin $t$ then contributes $|P_t|/\sum_{t'}|P_{t'}|$, which under uniform eye
sampling is $\approx n_t(n_t{-}1)/2$ normalized.

Under constant $n_t$ the two readings coincide ($|P_t|$ is the same for every
$t$). Under variable $n_t$ they differ — the bin with the most close pairs
either gets the same voice as every other (uniform) or dominates the average
(pair-count). Both are consistent across-bin weightings for the LOTC, and §3
develops the bias/variance tradeoff between them.

## A.8 When the corrections are a no-op

- **Extension 1.** If $n_t$ is constant across time bins, the uniform and
  pair-count directions coincide (the two readings of §A.7 give the same
  weight per bin), and Extension 1 reduces to the identity.
- **Extension 2.** The correction is the identity whenever (A2) holds at
  the second moment: $\mathbb E_t[r^2(t,e)]$ independent of $e$ makes
  $\mathbb E_{e\sim D}\!\big[\mathbb E_t[r^2]\big]$ the same for every $D$,
  so the close-pair second moment converges to the right
  $\mathrm{Var}_\text{total}$ regardless of its $p^2$ sampling. This
  reproduces McFarland's stated regime. Note however that even under (A2),
  $\mathrm{Var}_\text{PSTH}^D$ *does* depend on $D$ through the eye-
  distribution-spread of the PSTH integrand, so the Direction-1 and
  Direction-2 targets do not coincide in $1-\alpha$; their gap is the
  fixation-scale spatial-structure measure of §4.3. The
  `test_homogeneous_mask_correction_is_noop_for_full_target` test confirms
  that under (A2) (`flat` mask), `target='naive'` and `target='full'` agree
  on $1-\alpha^p$ while `target='central'` recovers $1-\alpha^{p^2}$.

## A.9 Consistency: how $\mathrm{sd}[1-\hat\alpha]$ depends on $N$ and $T$, and the $[0,1]$ clipping bias

Section 2.4 noted that the seed-to-seed SEM of McFarland's estimator does
**not** shrink as $1/\sqrt{N}$ alone — an across-time-bin noise floor in $T$
kicks in once within-bin sampling is adequate, and at high SEM the
$[0,1]$ clipping of $\hat\alpha$ introduces a mean bias. This appendix
derives both effects on the unified flat-mask synthetic under (A1)+(A2)
and calibrates them empirically against the closed form. The code is
`fig_consistency.py`; the empirical sweep is cached to
`consistency_sweep.npz`.

### A.9.1 The across-time-bin floor

Take the flat mask ($M\equiv 1$), constant $n_t = N$, and deterministic
rates (no observation noise). The per-time-bin PSTH is
$\mathbb E_{e\sim p}[r(t,e)] = \mu_0 + G_t$ with

$$
G_t = \int s_t(e)\,p(e)\,de.
$$

By (A2) the $\{G_t\}_{t=1}^{T}$ are i.i.d. zero-mean across time bins with

$$
V_p \;\equiv\; \mathbb E\,G_t^2
 \;=\; \iint K(e_1-e_2)\,p(e_1)\,p(e_2)\,de_1\,de_2
 \;=\; \frac{\tau^2\,\ell^2}{\ell^2+2\sigma_e^2}
 \;=\; \alpha^*\,\tau^2,
\qquad
\alpha^* \equiv \frac{\ell^2}{\ell^2+2\sigma_e^2}.
$$

In the large-$N$ limit the close-pair pool ($\sim T\cdot N(N{-}1)/2$ pairs)
makes $\widehat C_{\text{rate}}\to \tau^2$ deterministically, so the
estimator collapses to

$$
\hat\alpha \;\approx\; \frac{1}{\tau^2}\cdot\frac{1}{T-1}\sum_t\!(G_t-\bar G)^2,
$$

the sample variance of $T$ i.i.d. Gaussians divided by their (assumed
known) population variance scaling. Its sampling variance is

$$
\mathrm{Var}[\hat\alpha]
 \;=\; \frac{1}{\tau^4}\cdot\frac{2\,V_p^2}{T-1}
 \;=\; \frac{2\,{\alpha^*}^2}{T-1},
$$

giving the **across-time-bin SEM floor**

$$
\lim_{N\to\infty} \mathrm{sd}[1-\hat\alpha]
 \;=\; \alpha^*\,\sqrt{\frac{2}{T-1}},
\tag{A6.1}
$$

a finite limit in $T$ alone. At $\ell=\sigma_e$ (so $\alpha^*=1/3$) and
$T=100$, (A6.1) gives $\mathrm{sd}\approx 0.0474$, in agreement with the
leveling-off observed in §2.4.

### A.9.2 Within-bin contribution and the empirical decomposition

At finite $N$, the close-pair estimator's local fluctuations and the
finite-trial sampling of $G_t$ each contribute a within-bin noise term
that scales as $1/\sqrt N$ at fixed $T$. Treating the across-time-bin and
within-bin contributions as approximately independent,

$$
\mathrm{sd}[1-\hat\alpha]
 \;\approx\;
 \sqrt{\;\mathrm{sd}_{\text{within}}^2(N) \;+\; \mathrm{sd}_{\text{floor}}^2(T)\;},
\tag{A6.2}
$$

so adding trials drives the SD down toward the floor but does not pierce
it. Fig. A6A shows the heatmap of empirical $\mathrm{sd}[1-\hat\alpha]$
over $(N,T)$ at $\ell=\sigma_e$: the right edge (large $T$, $\mathrm{sd}\approx
0.04$) is dominated by the floor; the left edge (small $T$) is dominated
by the floor as well — across the whole grid the floor sets the bottom
of the achievable SD. Fig. A6C shows the same data sliced at fixed $N$:
the $N=800$ points sit on the analytical floor curve $\alpha^*\sqrt{2/(T-1)}$
at large $T$ and rise above it at small $T$ where within-bin noise
becomes significant relative to the floor.

The practical implication is that **for a target SEM on $1-\hat\alpha$,
more time bins is the only binding knob** once trials/bin is large enough
to drop within-bin noise below the floor. In `fixRSVP` with $T\sim 100$
post-fix bins, $\alpha^*=1/3$ gives a floor of $\approx 0.047$, comparable
to the cross-cell variability of the population $1-\alpha$ estimate.

The §2.3 stimulus-frame caveat applies here: in real `fixRSVP` adjacent
analysis bins inside a single $20$ Hz stimulus-frame interval share the
same field draw, so the effective $T$ for (A6.1) is
$\#\text{fixations}\times\#\text{stim-frames per fixation}$ rather than
the raw bin count. With a $50$ ms stimulus-frame period and
$\sim 6$–$12$ post-fix bins per fixation (depending on bin width), the
effective $T$ is reduced by the bins-per-frame multiplicity and the floor
correspondingly rises. We do not propagate this correction through the
real-data numbers in §4.5 — those are robust to it at the population
median (Δ $1-\alpha = -0.022$) — but it matters for per-cell SEM and is
the operative reason the within-stimulus-frame reliability question is
flagged as a future direction (§2.3).

### A.9.3 Boundary-clipping bias

`estimators.py:252` clips $\hat\alpha$ to $[0,1]$ before reporting
$1-\hat\alpha$. The clip is sensible — $\alpha$ is a fraction and the
ratio of two variance estimates can excursion outside — but it
introduces a bias on the mean whenever the sampling SD $\sigma_\alpha$ is
comparable to the distance from $\alpha^*$ to the nearest boundary.

Treating $\hat\alpha\sim\mathcal N(\alpha^*, \sigma_\alpha^2)$ and clipping
to $[0,1]$, the post-clip mean is the truncated-Gaussian moment

$$
\mathbb E\!\left[\mathrm{clip}(\hat\alpha)\right]
 = \alpha^*\bigl[\Phi(z_1)-\Phi(z_0)\bigr]
 \;+\; \sigma_\alpha\bigl[\varphi(z_0)-\varphi(z_1)\bigr]
 \;+\; \bigl[1-\Phi(z_1)\bigr],
\tag{A6.3}
$$

with $z_0 = -\alpha^*/\sigma_\alpha$, $z_1 = (1-\alpha^*)/\sigma_\alpha$, and
$\varphi,\Phi$ the standard normal pdf/cdf. The sign of the bias on
$1-\hat\alpha$ is set by which boundary is closer:

- **Small $\ell$** ($\alpha^*\to 0$, $1-\alpha^*\to 1$): the lower tail of
  $\hat\alpha$ is clipped up to $0$, so $\mathbb E[\mathrm{clip}(\hat\alpha)]>\alpha^*$
  and $1-\hat\alpha$ is biased **down**.
- **Large $\ell$** ($\alpha^*\to 1$, $1-\alpha^*\to 0$): the upper tail of
  $\hat\alpha$ is clipped down to $1$, so $\mathbb E[\mathrm{clip}(\hat\alpha)]<\alpha^*$
  and $1-\hat\alpha$ is biased **up**.

Both pull $1-\hat\alpha$ toward the interior $1/2$. Fig. A6B verifies
(A6.3) on the small-$\ell$ side: at $\ell=0.3\sigma_e$ ($\alpha^*=0.043$),
each $(N,T)$ cell produces an $(\mathrm{sd},\mathrm{bias})$ pair, and the
empirical points sit on the analytical curve as $\sigma_\alpha$ is varied.
The bias reaches $\sim -0.08$ at $\sigma_\alpha\approx 0.3$ — small in
absolute terms, but a $9\%$ shift relative to the true $1-\alpha^*=0.957$.
The same effect explains the saturation of $1-\hat\alpha$ at $0$ for
eccentric-mask cells (§4.1, Fig. 3), where the unclipped close-pair $\hat\alpha$
excursions above $1$ from the inflated-$C_\text{rate}$ side of the naive
inconsistency.

![**Figure A6 — SEM and $[0,1]$ clipping of $1-\hat\alpha$ on the
flat-mask synthetic.** **(A)** Empirical seed-to-seed $\mathrm{sd}[1-\hat\alpha]$
over a 4×4 $(N,T)$ grid at $\ell=\sigma_e$ (10 seeds per cell, deterministic
rates, `target='naive'`, threshold 0.05). The T-floor
$\alpha^*\sqrt{2/(T-1)}$ is printed below the panel; empirical SD shrinks
toward it as $N$ grows and is bounded below by it as $T$ shrinks. **(B)**
Boundary clipping at $\ell=0.3\sigma_e$ ($\alpha^*=0.043$,
$1-\alpha^*=0.957$): bias of $1-\hat\alpha$ from truth, one marker per
$(N,T)$ cell, coloured by $T$. The dashed curve is the analytical
truncated-Gaussian prediction (A6.3). **(C)** $\mathrm{sd}[1-\hat\alpha]$
vs $T$ at $\ell=\sigma_e$ for $N\in\{100,200,400,800\}$. The dashed line is
the closed-form floor $\alpha^*\sqrt{2/(T-1)}$; the large-$N$ points sit
on it.](figures/fig_consistency.png)

## A.10 The PSTH covariance estimator: McFarland M6 vs bagged split-half

The §1 estimators of $C_{\text{psth}}$ and $C_{\text{rate}}$ both rest on the
cross-trial trick of (4) — distinct-trial products at the same time bin have
independent observation noise, so the average product converges to the
underlying rate product with the noise removed. The close-pair estimator (8) is
one specific arrangement of this trick (pairs restricted to $\Delta e<\varepsilon$).
The PSTH-covariance estimator (6) is another (pairs unrestricted on $\Delta e$).
Two computationally distinct estimators target (6); this appendix walks through
why the naive PSTH covariance fails, names the two distinct-trial estimators
that fix it, and records the choice this writeup makes.

### A.10.1 Why the naive PSTH covariance is biased — and worse on off-diagonals

The textbook estimator of $C_{\text{psth}}$ is the across-time-bin covariance of
the per-bin trial-mean count

$$
\hat\mu_m(t) \;=\; \frac{1}{n_t}\sum_{i=1}^{n_t} Y_m^i(t)
 \;=\; \mathbb E[Y_m\mid t] \;+\; \underbrace{\frac{1}{n_t}\sum_i \xi_m^i(t)}_{\text{trial-mean noise}},
$$

where $\xi_m^i(t) = Y_m^i(t) - \mathbb E[Y_m\mid t]$ collects every source of
trial-to-trial variability at fixed time bin $t$: single-cell Poisson, the FEM
modulation $r_m(t,e)-\mathbb E_e[r_m\mid t]$, and any shared latent that
couples cells on the same trial. Taking $\mathrm{Cov}_t(\hat\mu_m, \hat\mu_n)$
gives, in expectation,

$$
\mathbb E\!\big[\widehat{\mathrm{Cov}_t}(\hat\mu_m,\hat\mu_n)\big]
 \;=\; \mathrm{Cov}_t\!\big(\mathbb E[Y_m\mid t],\, \mathbb E[Y_n\mid t]\big)
 \;+\; \mathbb E_t\!\left[\frac{\mathrm{Cov}\!\big(\xi_m,\xi_n\mid t\big)}{n_t}\right].
\tag{A7.1}
$$

The first term is the PSTH covariance we want. The second term is a
finite-trial *contamination* by the within-bin same-trial noise covariance,
attenuated by $1/n_t$ but not gone. It has two distinct failure modes:

- **On the diagonal** $(m=n)$, the second term is
  $\mathbb E_t[\mathrm{Var}(\xi_m\mid t)/n_t]$ — the familiar finite-trial
  inflation of any sample variance. McFarland and the older literature
  (Sahani–Linden 2003) routinely debias it via a within-bin variance subtraction
  $- MS_{\text{within}}/n_0$.
- **On the off-diagonal** $(m\neq n)$ the second term is
  $\mathbb E_t[\mathrm{Cov}(\xi_m,\xi_n\mid t)/n_t]$ — *the simultaneous
  cross-cell noise correlation*, attenuated by $1/n_t$ but present in
  expectation. This is the quantity the noise-correlation analysis is trying
  to measure (it shows up in $C_{\text{noise}}^{\text{corr}} = C_{\text{total}}
  - C_{\text{rate}}$). If $C_{\text{psth}}$ silently carries an attenuated
  copy of it, $C_{\text{noise}}^{\text{corr}}$ double-counts. The
  Sahani–Linden diagonal subtraction does **not** address this; off-diagonal
  debiasing needs an estimator that handles all matrix elements uniformly.

This is what the cross-trial trick fixes by construction. Any estimator that
averages products of *distinct* trials at the same bin — $Y_m^i(t) Y_n^j(t)$
with $i\neq j$ — has, in expectation, $\mathbb E[Y_m^i Y_n^j\mid e_i,e_j]
= r_m(t,e_i)\,r_n(t,e_j)$: trial $i$'s noise (single-cell Poisson *and*
same-trial cross-cell noise) is independent of trial $j$'s noise and drops out
of the cross-trial expectation. The diagonal and off-diagonal biases of (A7.1)
both vanish in the same step.

### A.10.2 Two distinct-trial estimators of $C_{\text{psth}}$

Once the requirement is "average distinct-trial products at the same bin",
two natural computational arrangements present themselves.

**McFarland M6 / M12 — all-distinct-pair second moment.** Enumerate every
$i<j$ pair at each bin $t$, weight each pair by $w_i w_j$ with the per-trial
importance weight $w_i = q(e_i)/p(e_i)$ for target distribution $q$, sum, and
subtract $\bar Y_m \bar Y_n$ taken under the matching weighted mean:

$$
\widehat{C_{\text{psth}}}^{\;\text{M6}}_{mn}
 \;=\;
 \frac{\sum_t \sum_{i<j} w_i w_j\, Y_m^i(t)\, Y_n^j(t)}{\sum_t \sum_{i<j} w_i w_j}
 \;-\; \bar Y_m\,\bar Y_n .
\tag{A7.2}
$$

This is McFarland's Eq. 6 (single cell, diagonal) and Eq. 12 (cross cell)
verbatim. It is **identical in structure** to the close-pair estimator (8)
with the $\Delta e<\varepsilon$ filter removed: $C_{\text{rate}}$ and
$C_{\text{psth}}$ are the same estimator at the two ends of the $\Delta e$
axis of the conditional second-moment surface $C_{\text{eye}}(\Delta e)$ —
$C_{\text{rate}}$ at its $\Delta e\to 0$ intercept, $C_{\text{psth}}$ at its
eye-distribution-marginalized asymptote. Implemented per bin via the algebraic
identity $\sum_{i<j} w_i w_j\, S_i\otimes S_j
 = \tfrac12\big[(\sum_i w_i S_i)\otimes(\sum_i w_i S_i) - \sum_i w_i^2\, S_i\otimes S_i\big]$
to avoid materialising the $O(\sum_t n_t^2)$ pair tensor (`estimators._all_pairs_second_moment`).

**Bagged split-half.** At each bin $t$, randomly partition the (weighted)
trials into halves $A$ and $B$; form the per-half PSTHs
$\hat\mu^{A}_m(t), \hat\mu^{B}_n(t)$ as the $w$-weighted trial means within
each half; take their across-bin cross-covariance; average over $n_{\text{boot}}$
independent random partitions. The cross-half product
$\hat\mu^{A}_m(t)\,\hat\mu^{B}_n(t)$ at bin $t$ is, by construction,
$(|A||B|)^{-1}\sum_{i\in A,\, j\in B} Y_m^i Y_n^j$ — a *subset* of (A7.2)'s
$i\neq j$ pairs, restricted to cross-half pairs. Each random partition uses
about half the available distinct-trial pairs; bagging over $n_{\text{boot}}$
partitions converges in expectation to the full M6 estimator
(`estimators._split_half_psth_cov`).

The two are **the same estimator family** — both target (6) by averaging
distinct-trial products. They differ only in computational form. McFarland M6
is the $n_{\text{boot}}\to\infty$ limit of bagged split-half, evaluated
deterministically as a closed-form sum.

### A.10.3 Tradeoffs

| | McFarland M6 (A7.2) | bagged split-half |
|---|---|---|
| bias | unbiased | unbiased in expectation |
| variance (fixed data) | floor — uses every distinct pair once | floor $+\;O(1/n_{\text{boot}})$ bootstrap penalty |
| determinism | exact (no seed) | seed-dependent |
| conceptual unity with $C_{\text{rate}}$ | same code path; $C_{\text{rate}}$ is the $\Delta e\to 0$ restriction of (A7.2) | parallel pipeline; the connection to $C_{\text{rate}}$ is via expectations, not code |
| importance reweighting | per-pair $w_i w_j$ (closed-form) | per-trial $w_i$ via the within-half weighted mean |
| compute | $O\!\big(\sum_t n_t\,C^2\big)$ via the per-bin identity | $O(n_{\text{boot}}\cdot T\cdot C^2)$ |
| side benefits | none | natural per-cell SEM across $n_{\text{boot}}$ bootstraps |

The variance ordering follows from M6 being the $n_{\text{boot}}\to\infty$
limit of split-half: the bootstrap estimator inherits M6's bias and adds an
$O(1/n_{\text{boot}})$ penalty from the finite partition count. The
conceptual-unity row is the structural difference: in the M6 path,
$C_{\text{rate}}$ and $C_{\text{psth}}$ flow out of the *same* function with
or without the $\Delta e$ filter; in the split-half path, $C_{\text{rate}}$
and $C_{\text{psth}}$ are computed by genuinely different code that happens
to target the same population quantities.

### A.10.4 Our choice — McFarland M6

We use the McFarland M6 estimator (A7.2) for every $C_{\text{psth}}$ reported
in this note (`decompose(cpsth_method='mcfarland')`, the default). The
deciding factor is the conceptual unification: the writeup's central story —
"the close-pair restriction in (8) silently changes the eye-position
distribution from $p$ to $p^2$, and importance reweighting restores
consistency" — reads more directly when $C_{\text{rate}}$ and
$C_{\text{psth}}$ are the same estimator at two operating points than when
they are two estimators that happen to coincide in expectation. Section §1.5's
$(w_t, q)$ table, in particular, becomes a single statement about which pairs
the all-pairs second moment averages and how it weights them, applied
identically to $C_{\text{rate}}$ and $C_{\text{psth}}$.

The bagged split-half implementation is retained at
`estimators._split_half_psth_cov` and reachable via
`decompose(cpsth_method='split_half')`. We do not lose anything by keeping it
— the two estimators agree within bootstrap noise at our $(N, T, C)$ scales
— and we preserve the option to fall back if its side benefits (natural
per-cell SEM via the bootstrap distribution, or memory advantage at very
large $C$ where the per-bin pair-sum identity in M6 starts to dominate) become
decisive on a future dataset.

The production pipeline (`VisionCore/covariance.py`,
`bagged_split_half_psth_covariance`) continues to use bagged split-half for
$C_{\text{psth}}$; that is a pre-existing implementation choice that this
note does not propose to change. The numerical agreement between the two
estimators on the same data (within bootstrap noise) means the writeup's
empirical claims transfer directly across the boundary between this folder's
M6-based decomposition and the production pipeline.

---

*Reproduce all figures and tests (from this folder):*

```bash
uv run python fig_mechanism.py
uv run python fig_distribution_truth.py
uv run python fig_sanity_check.py
uv run python compute_weighting_data.py     # Ext-1 real-data driver (cache-only, all sessions)
uv run python fig_weighting_bias.py         # Ext-1 cross-cell weighting bias (Fig. 1)
uv run python fig_recovery.py               # matched estimators recover analytical 1-α (Fig. 3)
uv run python fig_trajectory.py             # multi-bin trajectory extension (Fig. 4)
uv run python fig_consistency.py            # parallel sweep; cached to .npz
uv run python generate_realdata.py          # cache-only; --recompute to rebuild (Fig. 5)
uv run --with pytest pytest test_estimators.py -q
```

*Build this note to a self-contained, offline HTML (math via MathML, images
inlined):*

```bash
pandoc writeup.md -s --mathml --self-contained \
  --lua-filter=number-eqs.lua -o writeup.html
```

The `number-eqs.lua` filter restores equation numbers: pandoc's MathML writer
drops amsmath `\tag{}`, so without it every `(N)` call-out in the prose points
at an unnumbered equation.
