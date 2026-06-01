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
   across time bins; for the LOTC to hold term-by-term, the PSTH covariance and
   the mean entering the rate-variance subtraction must use the same
   pair-count weighting. Validated against closed-form synthetic ground truth
   with variable per-time-bin trial counts.
2. **Eye-position-distribution matching** under a non-homogeneous stimulus.
   Close pairs at threshold $\Delta e<\varepsilon$ are sampled in proportion
   to the *squared* eye-position density $p(e)^2$, while the total covariance
   and the PSTH covariance are over $p(e)$. We restore consistency with an
   importance-reweighted estimator that has the target eye distribution as a
   parameter, with two principled choices ($p$ and $p^2$). Their gap measures
   whether the rate has spatial structure on the fixation scale — non-zero
   even under (A2) when $\ell\sim\sigma$.

Sections 1 and 2 set up McFarland's estimator and the synthetic model that
breaks both assumptions. Sections 3 and 4 develop and validate one extension
each on the synthetic. Section 5 reports the consequences on synthetic and
real data. Section 6 records the production-pipeline state — Extension 1 has
already been integrated; Extension 2 is implemented in this folder and gated.

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
**(A)** Eye distribution $p(e) = \mathcal N(0,\sigma^2 I)$, $\sigma=0.15^\circ$;
dashed rings at $1\sigma,\,2\sigma$.
**(B)** One draw of the stationary Gaussian random field $s_t(e)$ with kernel
$K(\delta)=\tau^2\exp(-\lVert\delta\rVert^2/(2\ell^2))$ at $\ell=0.5\sigma$
(intentionally short to make spatial structure visible).
**(C)** The four spatial masks $M_c(e)$ — the (A2) switch — tiled into one
square: `flat`, `central`, `eccentric`, `linear`.
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
  is the rate map at time bin $t$. Independent time bins give the standard
  cross-trial cancellation of observation noise. At any fixed $e$,
  $s_t(e)\sim\mathcal N(0,\tau^2)$ across time bins — the across-time-bin
  distribution is independent of $e$, so the *field component* is (A2) by
  construction.
- $\alpha(t)$ — a per-time-bin amplitude envelope. Default $\alpha\equiv 1$.
  When the synthetic targets Extension 1 (§3) we set $\alpha(t)$ to decay
  across time bins, mirroring `fixRSVP`'s onset transients (high amplitude in
  early, high-$n_t$ time bins). Without correlation between $\alpha$ and
  $n_t$, the Extension-1 bias washes out.
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

Four mask shapes span the relevant cases:

| mask kind | $M_c(e)$ | physical role |
|---|---|---|
| `flat` | $1$ | homogeneous stimulus — (A2) holds |
| `central` | $\exp\!\big(-\lVert e\rVert^2/(2\ell_M^2)\big)$ | windowed stimulus: cell sees the stimulus near fixation, only baseline in periphery |
| `eccentric` | $1 - \exp\!\big(-\lVert e\rVert^2/(2\ell_M^2)\big)$ | the bounded complement — cell suppressed at fixation, sees the stimulus when eye drifts |
| `linear` | $\tfrac12\big(1+\tanh(x/\ell_M)\big)$ | smooth bounded $x$-gradient — the simplest unidirectional non-homogeneity |

`central` is the workhorse stand-in for the windowed `fixRSVP` mechanism.
`flat` is the regime in which McFarland's estimator is unbiased; we use it
in §2.4 to sanity-check the estimator against a non-trivial analytical
$1-\alpha^p$ that the additive model could not provide. `eccentric` and
`linear` stress-test the corrected estimator on masks whose variance lives
where close pairs are rare or biased.

**Eye distribution.** Eyes are drawn i.i.d. $e\sim p=\mathcal N(0,\sigma^2 I)$
per (trial, time-bin), $\sigma=0.15^\circ$ (realistic fixational drift). For an
isotropic Gaussian $p$, the close-pair density $p(e)^2$ is *exactly*
$\mathcal N(0,\sigma^2/2\,I)$ — a tighter Gaussian with half the variance —
so the LOTC under $p$ or $p^2$ has a closed form (Appendix §A.5).

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
practical consequence (developed in §A.6) is that the effective $T$
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
(derivation in Appendix §A.5):

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
they differ only in finite-sample efficiency (§3.3).

For the `flat` mask and Gaussian $D, K$, the integral closes analytically:

$$
1-\alpha^p = \frac{2\sigma^2}{\ell^2 + 2\sigma^2},
\qquad
1-\alpha^{p^2} = \frac{\sigma^2}{\ell^2 + \sigma^2},
\tag{11}
$$

a single-parameter family in $\ell/\sigma$ that covers $(0,1)$. For
`central` (Gaussian) the closed form is also available; for `eccentric`
and `linear` we use Monte Carlo (4M-sample default, sampling noise
$\lesssim 10^{-3}$).

## 2.4 McFarland's estimator recovers analytical $1-\alpha^p$ under (A1)+(A2)

The unified model lets us test McFarland's estimator *in his native regime*
against a closed-form $1-\alpha^p$ that covers $(0,1)$ — something the
purely additive synthetic could not do, because there the only
(A2)-respecting profile was $F\equiv\mathrm{const}$, which forced
$1-\alpha=0$. Setting `kinds=['flat']` and varying $\ell/\sigma$ traces the
full $1-\alpha^p$ axis.

The empirical estimator is `decompose(target='naive', time_bin_weighting='pair_count')`
with constant $n_t$ — the same shape of close-pair estimator McFarland's
paper specifies. Under (A2), $\mathbb E_t[r^2(t,e)]$ is independent of $e$
(the field component is stationary, and the mask is constant), so
the close-pair second moment converges to $\mathbb E_t[r^2(t,e)]$ at *any*
$e$ — and in particular to the same value $\tau^2 + \mu_0^2$ regardless of
which density samples the close-pair midpoints. With $\bar Y \to \mu_0$,
$C_\text{rate}\to\tau^2 = \mathrm{Var}_\text{total}^p$ and $C_\text{psth}$
(split-half over $p$) $\to\tau^2\ell^2/(\ell^2+2\sigma^2)$, giving
$1-\alpha\to 2\sigma^2/(\ell^2+2\sigma^2) = 1-\alpha^p$. So McFarland is
unbiased for $1-\alpha^p$ — even though the close-pair density itself is
intrinsically $p^2$, not $p$.

Figure 0 confirms this empirically across an $\ell/\sigma$ sweep and shows
the analytical closed-form gap as a function of $\ell/\sigma$; the gap is
**non-zero** under (A2), peaking at $\ell\approx\sigma$. This is the
reframing §4.5 builds on: the gap is a fixation-scale spatial-structure
measure, not an (A2) test.

The estimator's *consistency* (how the seed-to-seed SEM depends on the
trials-per-time-bin $n$ **and** the number of time bins $T$) is more subtle than
$1/\sqrt{n}$: an across-time-bin noise floor at
$\sqrt{2\alpha^2/(T{-}1)}$ kicks in once within-bin sampling is
adequate, and at high SEM the $[0,1]$ clipping of $\alpha$ introduces a
mean bias. This is treated separately in Appendix §A.6.

![**Figure 0 — McFarland's estimator under (A1)+(A2).** **(A)** Empirical
`decompose(target='naive')` (red, mean$\pm$sd across 6 seeds) sits on the
analytical $1-\alpha^p$ curve (blue) across an $\ell/\sigma$ sweep covering
$(0,1)$. **(B)** Threshold robustness over a useful range. **(C)**
Closed-form $1-\alpha^p$, $1-\alpha^{p^2}$, and their gap vs $\ell/\sigma$.
**The gap is non-zero under (A2)**, vanishing only as $\ell\to 0$
(decorrelated rates, all FEM) or $\ell\to\infty$ (uniform field, no FEM);
it peaks at $\ell/\sigma\approx 1.18$.](figures/fig_sanity_check.png)

---

# 3. Extension 1: pinning the $w_t$ column

## 3.1 The $w_t$ cells of McFarland's literal estimators

The $w_t$ column of the §1.5 table reads directly off the literal forms.

$C_\text{total}$, the LHS of the LOTC (1), is the sample variance of $Y^i(t)$
pooled over all $(i,t)$. With each sample contributing one unit of weight, bin
$t$ contributes $n_t$ units; the implicit across-bin weighting is **trial-count
$w_t \propto n_t$**.

$C_\text{psth}$ uses (6) — the cross-trial average of distinct-pair products at
the same time bin. McFarland's nested bracket $\langle\langle\,\cdot\,\rangle_{i\ne
j}\rangle_t$ averages within $t$ first and then averages bins uniformly: every
bin contributes **$w_t = 1/T$** regardless of how many pairs it contains. The
$\bar Y\bar Y^\top$ subtractor is the global mean over $(i, t)$ — trial-count
$w_t \propto n_t$.

$C_\text{rate}$ uses (8) — the same nested bracket restricted to close pairs
$\Delta e_{ij} < \varepsilon$. Same reading: second-moment term at
$w_t = 1/T$, $\bar Y\bar Y^\top$ subtractor at trial-count.

Three different across-bin weightings across the three estimators, and an
additional second-moment-vs-$\bar Y$ split inside each of $C_\text{psth}$
and $C_\text{rate}$. Under (A1) constant $n_t$ all four cells coincide (trial-count
and $1/T$ collapse to a constant under scaling) and the inconsistency is
invisible. Under variable $n_t$ they do not: the LOTC (1) fails term-by-term
because the three estimators no longer measure variance/covariance under the
same across-bin distribution.

## 3.2 Two consistent directions: uniform $1/T$ and pair-count $\propto n_t(n_t{-}1)/2$

Restoring term-by-term consistency means pinning the entire $w_t$ column — every
second-moment term and every $\bar Y$ subtractor in every estimator — to a single
across-bin weighting. Two principled choices are available.

**Uniform direction** ($w_t = 1/T$). Match McFarland's literal nested-bracket
reading: each fixation-aligned bin contributes equally regardless of $n_t$. The
$\bar Y\bar Y^\top$ subtractor (and $C_\text{total}$) are recomputed as
$\frac{1}{T}\sum_t \frac{1}{n_t}\sum_i Y^i(t)$ — within-bin mean first, then
uniform across bins. The resulting estimator answers "how does the variance
behave averaged equally over the analysis time bins" — the natural reading of
the LOTC LHS for a fixation-aligned analysis.

**Pair-count direction** ($w_t \propto n_t(n_t{-}1)/2$). Pool all distinct same-bin
pairs across $t$ into a single flat list and average uniformly per pair. Under
uniform eye sampling, bin $t$ contributes $n_t(n_t{-}1)/2$ close pairs, so its
implicit across-bin weight is $\propto n_t(n_t{-}1)/2$. The
$\bar Y\bar Y^\top$ subtractor and $C_\text{total}$ are recomputed with the
same per-bin total weight $\propto n_t(n_t{-}1)/2$.

Both directions are consistent — every cell of the $w_t$ column lands on a
single value, and the LOTC (1) holds term-by-term. They differ in efficiency.
The per-bin close-pair second-moment estimator has sampling variance
$\propto 1/|P_t|$ where $|P_t| \approx n_t(n_t{-}1)/2 \cdot \Pr[\Delta e<\varepsilon]$;
the inverse-variance-optimal across-bin combination weights bin $t$ by $|P_t|$
— exactly pair-count. The uniform direction gives every bin equal voice, the
natural reading of "average across analysis time bins", but pays a variance
penalty when $n_t$ varies sharply because low-$n_t$ bins enter with the same
weight as high-$n_t$ bins despite their much noisier per-bin estimates.

The same bias/variance tradeoff appears on the $q$ axis in §4.5: Direction 1
(full $p$) is the natural target but its unbounded $1/p$ close-pair weights
are noisy in the periphery; Direction 2 (central $p^2$) uses bounded weights
but reports a value shifted toward the close-pair density.

## 3.3 Synthetic validation and headline choice

The truth $1-\alpha$ is **invariant under time-bin weighting** in the unified
rate field (10): the envelope factor $\mathbb E_w[\alpha^2]$ cancels in the
ratio. Both consistent directions therefore target the same closed-form
$1-\alpha^p$. Pair the staircase $n_t$ ($15\to360$ across $T=100$ bins) with a
decaying envelope $\alpha(t)$ that concentrates amplitude in early high-$n_t$
bins (Fig. 1A) — the case that maximally separates the two directions in
finite samples.

![**Figure 1 — Two consistent $w_t$ directions on the (A2)-respecting flat
mask under variable $n_t$.** **(A)** The variable-$n_t$ staircase and the two
across-bin weight curves: uniform $w_t = 1/T$ (red) and pair-count $w_t\propto
n_t(n_t{-}1)/2$ (blue). Under constant $n_t$ both are flat; under variable
$n_t$ the pair-count weight strongly emphasizes early high-$n_t$ bins.
**(B)** Histograms of $1-\hat\alpha$ across seeds for both directions at
$\ell=\sigma$, on flat-mask synthetic with the staircase + envelope.
Closed-form truth $1-\alpha^p\approx 0.667$ (dashed). **Both directions are
unbiased**; pair-count has a tighter distribution than uniform. **(C)**
Continuous $\ell/\sigma$ sweep: across-seed SD of $1-\hat\alpha$ for the two
directions, with the closed-form $T$-floor $\alpha^*\sqrt{2/(T-1)}$
overlaid. The efficiency gap of uniform vs pair-count persists across spatial
scales.](figures/fig_time_bin_weighting.png)

Panel B confirms that the truth is $w_t$-invariant and that both directions
hit it — the difference between the two histograms is variance, not bias.
Panel C extends to a continuous $\ell/\sigma$ sweep: at every spatial scale,
uniform's across-seed SD exceeds pair-count's. The gap is driven by low-$n_t$
late bins entering uniform with the same weight as high-$n_t$ early bins
despite their noisier per-bin contribution.

**We headline the pair-count direction** for all reported $1-\alpha$ values,
with one explicit caveat: pair-count concentrates the close-pair signal on
early high-$n_t$ bins, so the reported value leans on the early-trial portion
of each fixation. The uniform direction is the more natural reading of
"average over fixation-aligned time", but its finite-sample variance under
variable $n_t$ is too high to use as the primary report. The same asymmetry
appears on the $q$ axis (§4.5): Direction 1 (full $p$) is headlined because
the population bias relative to the actual-viewing target is small, accepting
modest variance cost.

## 3.4 The shuffle-null fix in production

A useful diagnostic for any $w_t$ mismatch: under a *shuffle null* that
destroys the eye–spike coupling (random eye trajectories), the close-pair
estimator has no real eye dependence to exploit and $C_\text{rate}^{\text{shuf}}$
should converge to $C_\text{psth}$ at the same $w_t$. If $C_\text{rate}$ and
$C_\text{psth}$ are computed with *different* $w_t$, then
$D_z = f_z(C_\text{noise}^{\text{corr}}) - f_z(C_\text{noise}^{\text{uncorr}})$
deviates from zero under the null *purely* because of the weighting mismatch
— independent of any (A2) violation, and independent of which direction is
chosen as long as $C_\text{rate}$ and $C_\text{psth}$ both use it.

This is the bias diagnosed and fixed in `ryan/fig2/bias_diagnosis/`
($D_z^{\text{shuf}}: -0.0068 \to +0.0010$, $p: <10^{-4}\to 0.44$, after consistent
pair-count weighting). The pre-fix production pipeline lived in a
partially-pinned state — close-pair second moment at pair-count (pool-then-average),
$C_\text{psth}$ at uniform $1/T$, $\bar Y$ at trial-count — that the
post-fix pipeline cleaned by pinning all three to the pair-count direction.
§6.1 records the production state.

## 3.5 What the correction does and does not address

Extension 1 is *only* about the weighting *across* time bins. It is a no-op
when the two directions coincide — either by constant $n_t$ (the
intersection with (A1); §A.4) or by an envelope $\alpha(t)$ uncorrelated with
$n_t$. It does not address Extension 2's eye-position distribution mismatch,
which acts *within* time bins on the choice of $e$ used to average the rate
variance. The two are orthogonal: Extension 1 fills the $w_t$ column of the
§1.5 table; Extension 2 fills the $q$ column.

---

# 4. Extension 2: pinning the $q$ column

Section 3 filled the $w_t$ column of the §1.5 table by pinning all three
estimators to a single across-bin weighting. This section fills the $q$
column by pinning all three to a single across-eye distribution. The two
extensions are independent — §3 addressed (A1) failure (variable $n_t$); §4
addresses (A2) failure (non-homogeneous stimulus) — and the same
natural-vs-stable tradeoff appears on this axis: McFarland's literal close-pair
second moment is pinned to the close-pair conditional density $p^2$, the actual viewing
distribution is $p$, and §4.5 develops both consistent directions and their
bias/variance tradeoff.

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
confirms this geometrically and numerically and shows the resulting
direction of the naive estimator's bias on $1-\alpha$ across mask kinds.

![**Figure 2 — Close pairs sample the squared density $p(e)^2$, and this
shifts $1-\alpha$ in a mask-dependent direction.** **(A)** Analytical eye
density $p(e) = \mathcal N(0, \sigma^2 I)$ with $\sigma = 0.15^\circ$
(grayscale), with iso-density contours at $1,2\sigma$ for $p$ (solid) and
for the close-pair density $p^2 = \mathcal N(0, \sigma^2/2\,I)$ (dashed) at
each distribution's own characteristic scale — the dashed contours are
tighter (variance halved). **(B)** The $x$-marginal: the
close-pair distribution matches $\mathcal N(0,\sigma^2/2)$ (observed variance
ratio $\approx 0.5$). **(C)** Closed-form / 4M-MC $1-\alpha$ at $\ell=\sigma$,
$\ell_M=0.6\sigma$, for the unified rate-field model. Blue: the truth
$1-\alpha^p = 1 - I_{M,K,p} / (\tau^2\,\mathbb E_p[M^2])$. Red: the naive
estimator $1-\alpha^{\mathrm{naive}} = 1 - I_{M,K,p} / (\tau^2\,
\mathbb E_{p^2}[M^2])$, which retains the correct PSTH numerator but uses
the close-pair $\mathbb E_{p^2}[M^2]$ in the denominator. The bias is
upward for `central` (denominator inflated) and downward for `eccentric`
/ `linear` (denominator deflated).](figures/fig_mechanism.png)

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
the mechanical origin of the variance penalty discussed in §4.5.

**Equivalence to eye-position stratification.** Importance reweighting (15) is
equivalent to **stratifying by absolute eye position**: partition $e$ into
strata, estimate the close-pair second moment *within* each stratum (where
$p\approx$ const so $p^2\approx p$ locally and the local estimate is unbiased
for $\mathbb E[r^2\mid s]$), then aggregate strata with weights equal to the
target occupancy $q(s)$. As the strata shrink this is exactly (15); with
finite strata it pools pairs before weighting, trading a little resolution for
lower variance. This is the natural generalization of McFarland et al.: their
estimator conditions on time bin $t$ and on $\Delta e\approx0$; the correction
conditions *additionally* on absolute eye position.

## 4.5 Direction 1 vs Direction 2: tradeoff and fixation-scale spatial-structure measure

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
 = \frac{\sigma^2\,\ell^2}{(\ell^2 + 2\sigma^2)(\ell^2 + \sigma^2)},
\tag{16}
$$

which is **non-zero for finite $\ell$**, vanishing only as $\ell\to 0$
(decorrelated rates, all FEM under every $D$) or $\ell\to\infty$ (uniform
field, no FEM). The maximum is $\approx 0.17$ at $\ell/\sigma\approx 1.18$
(Fig. 0D). The gap therefore measures whether the cell's **rate has spatial
structure on the fixation scale**: small gap when the rate is essentially
constant within a fixation ($\ell\gg\sigma$) or essentially decorrelated
within a fixation ($\ell\ll\sigma$); large gap when the rate's spatial
scale and the fixation scale are comparable. Non-homogeneous masks
(`central`, `eccentric`, `linear`) **add** to this baseline; Fig. 4C plots
the empirical gap across mask kinds with the (A2) baseline overlaid.

The gap is still a useful empirical signal — it tells you that the *choice*
of eye-position distribution $D$ matters for the reported $1-\alpha$ — but
it is not a clean test of (A2).

## 4.6 Multi-bin eye trajectories: the production-setting extension

§4.4 framed the close-pair filter as a single-bin condition
$\lvert e_i-e_j\rvert<\varepsilon$ — and the §5.2 real-data analysis honours
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

This breaks the §4.4 importance-weight construction as written. The close-pair
density now lives on $\mathbb R^{2T}$ — the trajectory density $p(\tau)$
squared, restricted to $\lVert\tau_i-\tau_j\rVert_{\mathrm{RMS}}<\varepsilon$ — and
fitting a density in $\mathbb R^{2T}$ for typical $T\in[10,30]$ is the curse of
dimensionality. We *cannot* simply lift the §5.2 single-bin KDE to the
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
pair $k$, contributing $T$ per-bin positions to the pool). The §4.4 importance
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
constant, and the table above recovers §4.4's Direction 1 pair weight
$1/p_{\mathrm{centroid}}(c)$ and Direction 2 sample weight $\propto p_{\mathrm{centroid}}(c)$
exactly. Self-normalisation absorbs the normalising constant in numerator and
denominator.

**The flat-trajectory approximation.** For $\sigma_{\mathrm{drift}}>0$ the
estimator targets the *per-bin marginal* $p_{\mathrm{pb}}(e)=p_{\mathrm{centroid}}*\phi_{\sigma_{\mathrm{drift}}}(e)$
— a Gaussian-smoothed version of the centroid distribution. The "actual viewing
distribution" the estimator aims at is therefore $\mathcal N(0,\sigma_{\mathrm{traj}}^2 I)$
with $\sigma_{\mathrm{traj}}^2=\sigma^2+\sigma_{\mathrm{drift}}^2$, and the truth to
compare against is §4.4's closed form at $\sigma_{\mathrm{traj}}$
(`synthetic.ground_truth(kind, sqrt(sigma^2+sigma_drift^2), ell, ell_M)`). The
construction is *exact* in expectation in the flat limit; for non-zero drift
the residual comes from two sources:

1. **Centroid-vs-per-bin smoothing.** Evaluating the ratio at the trajectory
   centroid (a single 2-D point) discards within-window drift; the ratio at
   $c_i$ is biased by an amount that scales with $\sigma_{\mathrm{drift}}/\sigma$.
2. **Threshold inflation.** Because $E[\lVert\tau_i-\tau_j\rVert^2_{\mathrm{RMS}}]
   \approx d_{\mathrm{centroid}}^2 + 4\sigma_{\mathrm{drift}}^2$, the RMS-trajectory
   threshold $\varepsilon$ must grow with $\sigma_{\mathrm{drift}}$ to admit any
   close pairs at all; at large $\sigma_{\mathrm{drift}}$ the close-pair filter
   loses selectivity and the central-region concentration of the close-pair
   midpoint density weakens.

**Validation (Fig. 5).** A controlled synthetic with explicit
$\sigma_{\mathrm{drift}}$ knob (`synthetic.make_trajectory_session`) shows the
flat-limit recovery is sharp and the bias grows smoothly with
$\sigma_{\mathrm{drift}}/\sigma$. At $\sigma_{\mathrm{drift}}/\sigma=0$ the corrected
Directions 1 and 2 sit on their respective truths within seed noise (panel
E); at $\sigma_{\mathrm{drift}}/\sigma\sim 0.2$ — comparable to the operating
regime for fixational drift over a typical $t_{\mathrm{hist}}+t_{\mathrm{count}}$
window — the bias is small; at $\sigma_{\mathrm{drift}}/\sigma\sim 1$ the
trajectories are no longer "essentially flat" and the bias is visible but
bounded. Panels B–D show the §4.1 mechanism reappearing in the multi-bin
setting: $\hat p_{cp,\mathrm{marg}}$ is narrower than $\hat p_{\mathrm{marg}}$, and
their ratio peaks at the centre — the close pairs over-represent the
high-density region exactly as the single-bin §4.1 picture predicts.

![**Figure 5 — The trajectory-mode estimator and its validation.** **(A)**
Example trajectories ($\sigma_{\mathrm{drift}}=\sigma/4$) showing centroid scatter
plus per-bin fixational drift. **(B)** $\hat p_{\mathrm{marg}}(e)$, the 2-D KDE
fit on pooled per-bin positions of all samples. **(C)** $\hat p_{cp,\mathrm{marg}}(e)$,
the 2-D KDE fit on per-bin positions of close-pair midpoint trajectories —
narrower than (B), concentrated at the centre. **(D)** The ratio
$\hat p_{cp,\mathrm{marg}}/\hat p_{\mathrm{marg}}$ — the §4.1 distribution mismatch in
the multi-bin setting, peaking at the centre as expected. **(E)**
$\sigma_{\mathrm{drift}}$ sweep on a `flat` mask: matched Directions 1 and 2 sit on
their respective truths (dotted lines) in the flat limit; bias grows smoothly
with $\sigma_{\mathrm{drift}}/\sigma$; naive over-states.](figures/fig_trajectory.png)

The trajectory-mode estimator is the production-setting bridge for §5.2's
single-bin analysis: when the §6.2 production change lands, the same
`target ∈ {'naive','full','central'}` parameter selects the same three
behaviours; the §5.2 numbers are recovered as the
$\sigma_{\mathrm{drift}}\to 0$, $T=1$ limit; and the curse-of-dimensionality wall
that the multi-bin filter raised is sidestepped by replacing the trajectory
density with two 2-D KDEs evaluated at the trajectory centroid.

---

# 5. Consequences on synthetic and real data

## 5.1 The naive estimator's $1-\alpha$ failure (synthetic)

Pure-Poisson synthetic data (true noise correlation $0$, true Fano $1$) under
the §2 unified generator, with the three non-homogeneous masks, exposes the
naive estimator's bias most clearly on $1-\alpha$ (Fig. 3A):

- **$1-\alpha$** (panel A): the naive estimator **over-states** for central
  masks (close pairs over-represent the center, where $M^2$ is large) and
  **under-states** for eccentric masks — occasionally producing
  $C_\text{rate} \approx 0$ that clips $1-\alpha$ to $0$. The matched
  estimator (target $p$) lands on the identity line within seed noise.

The sign rule is set by where each cell's stimulus visibility lives: the
close-pair density $p^2(e)$ emphasizes the center, so a mask whose
$E_{p^2}[M^2]/E_p[M^2]>1$ (a **central** mask) over-states $C_\text{rate}$
and a mask whose ratio is $<1$ (an **eccentric** mask) under-states it.

The leak of this $1-\alpha$ bias into the **noise correlation** and **Fano
factor** is qualitatively present but quantitatively mild under the unified
model (Fig. 3B–C: naive median $|r|=0.015$ vs matched $0.010$; both Fano
medians $\approx 1.01$). The additive model in earlier drafts predicted much
larger leaks because the additive $bF(e)$ term shifted the *mean* of $r$
with $e$, generating a large cross-term in $\bar Y^2$. In the unified
multiplicative model the mean $\mathbb E_t[r(t,e)] = \mu_0$ stays
$e$-independent, so the cross-term vanishes in expectation and only the
$M^2$-ratio bias survives. The real-data findings in §5.2 (small population
shifts on Fano, $0.846\to 0.875$) are consistent with this milder regime.

![**Figure 3 — The naive (distribution-unmatched) estimator is biased on
$1-\alpha$ in a mask-dependent direction; its leak into noise correlation
and Fano is mild under the unified model.** Pure-Poisson synthetic (true
noise correlation $0$, true Fano $1$); "matched" is the $p$-target corrected
estimator. **(A)** $1-\alpha$ vs ground truth: naive (×) biased centrally
(low values) or oversaturated; matched (○) on identity. **(B)** Noise
correlation: both estimators close to $0$ (truth); naive slightly wider.
**(C)** Fano: both estimators near $1$ (truth); naive slightly biased
high.](figures/fig_naive_failure.png)

The matched estimator recovers each target's own closed-form decomposition
(`test_estimators.py::test_full_target_recovers_p_decomposition` and
`::test_central_target_recovers_p2_decomposition`), up to a small
finite-threshold smoothing that shrinks as the threshold shrinks. Figure 4
illustrates the recovery and the Direction-1/Direction-2 tradeoff.

![**Figure 4 — The matched estimator recovers ground truth and exposes the
Direction-1/Direction-2 tradeoff.** **(A)** `full` recovers the $p$
decomposition and `central` the $p^2$ decomposition (identity line). **(B)**
For an eccentric-modulated cell, Direction 1's unbounded $1/p$ weights
make $1-\alpha$ noisy as the threshold shrinks; Direction 2 is stable.
**(C)** The full-vs-central gap across mask kinds (unified random-field
synthetic, $\ell=\sigma$). The dashed line is the closed-form (A2)
baseline ($\approx 0.17$) — the gap is non-zero even for `flat` because
the field has spatial structure on the fixation scale. Non-homogeneous
masks (`central`, `eccentric`, `linear`) shift the gap relative to the
baseline; the gap is a measure of *fixation-scale spatial structure*, not
an (A2) test.](figures/fig_correction.png)

## 5.2 Real-data consequences (`fixRSVP`, cache-only)

We applied the matched estimator to the real `fixRSVP` recordings, cache-only
(no GPU, no model inference; `generate_realdata.py` reads the Figure-4 cache
of trial-aligned spikes and real eye trajectories). $1-\alpha$ and the Fano
factor are computed on the real spikes with each cell's own validity mask
(reproducing the Figure-2 per-cell $1-\alpha$ at the median); the
eye-position density is a Gaussian KDE of the measured fixational positions.
This implementation uses a *single-bin* close-pair filter (the eye position
at one analysis time bin per sample), so the §4.4 importance weights apply
without modification; the multi-bin trajectory extension required by the
production filter (§4.6) is folded into the gated §6.2 pipeline change.
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
  a population **median of 0.089** with a tail beyond $0.3$ (Fig. 6B).
  Under the unified random-field model the gap is non-zero under (A2)
  itself when $\ell\sim\sigma$ (Eq. 16) — peaking at $\approx 0.17$ — so
  gap $= 0.089$ is evidence that the cells' rate maps have spatial
  structure on the fixation scale, which is *expected* for any cell with a
  finite spatial RF, not necessarily that (A2) is violated. What the gap
  *does* say is that the **choice** of eye-position distribution $D$
  matters for the reported $1-\alpha$ at this scale: a $\pm 0.09$ swing
  between Direction 1 and Direction 2 is the order of the
  Direction-1-vs-naive bias we found above ($-0.022$).
- **The Fano factor shifts modestly** under matching (median
  $0.846\to0.875$, $+3\%$), consistent with the synthetic prediction that
  the Fano factor inherits the rate-variance distribution mismatch
  (Fig. 3C); per-cell shifts are larger.

![**Figure 6 — The correction on real data (397 good cells, cache-only).**
**(A)** $1-\alpha$ on real spikes: Direction 1 (blue) tracks the naive
estimate closely (median shift $-0.022$), while Direction 2 (red) is
systematically lower. **(B)** The full-vs-central gap — a measure of the
rate's spatial structure on the fixation scale — has median $0.089$ with a
heavy tail. **(C)** Fano
factor: naive vs matched, a modest median shift with larger per-cell
changes.](figures/fig_realdata.png)

The **noise-correlation** consequence on real spikes is small at the
population level — both on the unified synthetic (Fig. 3B: naive vs matched
median $|r|$ differ by $\approx 0.005$) and consistent with the modest Fano
shift here ($0.846\to 0.875$). Quantifying it directly on real spike pairs
requires either the full windowed pipeline or careful joint-pair validity
masking; because it shares the same pipeline change, it is folded into the
gated Figure-2 fix (§6) rather than approximated here.

---

# 6. The current pipeline: state and proposed production change

The covariance machinery and caches are shared by Figures 2–4, so a pipeline
change has broad blast radius. This section consolidates the
production-pipeline state for each extension.

## 6.1 Extension 1 — already integrated

Consistent pair-count time-bin weighting has been integrated into the production
pipeline (`VisionCore/covariance.py`):

- `estimate_rate_covariance` — pair-count-weighted $\bar Y$ matching the
  close-pair second moment, pinning the $\bar Y$ cell of the §1.5 table to
  the pair-count direction (§3.2).
- `bagged_split_half_psth_covariance` — `weighting` parameter, default
  `'pair_count'`, pinning $C_\text{psth}$ to the pair-count direction.

Together with the trial-count $C_\text{total}$ already being recomputed at
the same per-sample weight inside the production estimator, this pins the
full $w_t$ column to the pair-count direction (§3.2).

On the Allen 2022-04-13 session (49 cells, 3667 windows) the fix moved the
shuffle-null $D_z$ from $-0.0068$ ($p<10^{-4}$) to $+0.0010$ ($p=0.44$); the
real-data $D_z$ shifted only $-0.0855 \to -0.0819$ (signal-to-null ratio
$12.5\times \to 83.7\times$), so the scientific conclusion was preserved.
Diagnosis and validation are recorded in
`ryan/fig2/bias_diagnosis/FINAL_REPORT.md`.

## 6.2 Extension 2 — implemented here, production gated

Eye-position-distribution matching is implemented and TDD-validated in this
folder, for both the single-bin (§4.4) and multi-bin trajectory (§4.6)
filters:

- `estimators.decompose(target=...)` — single-bin close-pair filter.
  `naive` reproduces the existing pipeline; `full` is Direction 1 (the
  actual-viewing $p$); `central` is Direction 2 ($p^2$).
- `estimators.decompose_trajectory(target=...)` — multi-bin RMS-trajectory
  close-pair filter (matching `VisionCore/covariance.py`), with the
  pooled-per-bin KDE importance weights of §4.6.
- `test_estimators.py` — 15 tests covering correctness (recovery of the
  closed-form decompositions under each target, finite-threshold-bias
  shrinkage), stability (Direction 2 stabler than Direction 1 for eccentric
  cells), Poisson cancellation (Fano $\to 1$), the pipeline-match
  (`naive` ↔ `pipeline_one_minus_alpha`), Extension 1 (variable-$n_t$
  uniform vs pair-count weighting), and the §4.6 multi-bin extension
  (flat-limit recovery, moderate-drift recovery, strong-drift bias
  documentation, naive bias on a centrally-modulated cell).

The proposed production change is to add the `target` argument to
`estimate_rate_covariance` (close-pair weight $q/p^2$ derived from the §4.6
pooled-per-bin KDE ratio at the trajectory centroid) and a corresponding
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

## A.3 Two readings of the nested-bracket close-pair estimator

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

## A.4 When the corrections are a no-op

- **Extension 1.** If $n_t$ is constant across time bins, the uniform and
  pair-count directions coincide (the two readings of §A.3 give the same
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
  fixation-scale spatial-structure measure of §4.5. The
  `test_homogeneous_mask_correction_is_noop_for_full_target` test confirms
  that under (A2) (`flat` mask), `target='naive'` and `target='full'` agree
  on $1-\alpha^p$ while `target='central'` recovers $1-\alpha^{p^2}$.

## A.5 Closed-form decomposition for the unified rate field

With the unified rate equation (9), the across-time-bin and across-eye
distributions decouple by linearity. Write $X_t(e) = M(e)\,\alpha(t)\,
s_t(e)$, so $r = \mu_0 + X$. The field is zero-mean, $\mathbb E[s_t(e)] = 0$,
with covariance $K(\delta) = \tau^2\exp(-\lVert\delta\rVert^2/(2\ell^2))$
and independent draws across time bins.

**Mean.** $\mathbb E_{t,e\sim D}[r] = \mu_0$ for any $w, D$ (the field is
zero-mean and the mean is constant in $e$).

**Total variance.** $\mathrm{Var}_\text{total}^{D,w} = \mathbb E[X^2]
= \mathbb E_w[\alpha^2]\,\mathbb E_D[M^2]\,\mathbb E_s[s_t(e)^2]
= \mathbb E_w[\alpha^2]\,\tau^2\,\mathbb E_D[M^2]$.

**PSTH variance.** Let $G_t^D = \mathbb E_{e\sim D}[M(e)\,s_t(e)]$. Then
$\mathbb E_{e\sim D}[r_t(e)] = \mu_0 + \alpha(t)\,G_t^D$ and

$G_t^D$ is i.i.d. across time bins with $\mathbb E[G_t^D] = 0$ and
$\mathbb E[(G_t^D)^2] = I_{M,K,D}$ (see below). In the large-$T$ limit

$$
\mathrm{Var}_\text{PSTH}^{D,w}
 = \mathbb E_w[\alpha^2]\,I_{M,K,D},
$$

where

$$
I_{M,K,D} = \iint M(e_1)\,M(e_2)\,K(e_1{-}e_2)\,D(e_1)\,D(e_2)\,de_1\,de_2.
$$

For the `flat` mask ($M \equiv 1$) the integral reduces to
$\mathbb E_u[K(u)]$ with $u = e_1 - e_2$. If $D = \mathcal N(0,\sigma^2 I)$
in $\mathbb R^2$, then $u \sim \mathcal N(0, 2\sigma^2 I)$ and the
Gaussian-Gaussian integral evaluates to

$$
\mathbb E_{u\sim\mathcal N(0, 2\sigma^2 I)}\!\big[\tau^2 e^{-\|u\|^2/(2\ell^2)}\big]
 = \tau^2 \cdot \frac{\ell^2}{\ell^2 + 2\sigma^2}.
$$

giving the `flat` mask closed forms

$$
1-\alpha^p = \frac{2\sigma^2}{\ell^2 + 2\sigma^2},
\qquad
1-\alpha^{p^2} = \frac{\sigma^2}{\ell^2 + \sigma^2},
$$

and the (A2)-respecting gap

$$
(1{-}\alpha^p) - (1{-}\alpha^{p^2})
 = \frac{\sigma^2\,\ell^2}{(\ell^2 + 2\sigma^2)(\ell^2 + \sigma^2)}.
$$

The maximum is at $\ell/\sigma = \sqrt{2}^{1/2} \approx 1.19$ (taking the
derivative in $\ell^2$ and setting to zero).

For the **central** mask $M(e) = \exp(-\lVert e\rVert^2/(2\ell_M^2))$,
the products $M(e_1)M(e_2)D(e_1)D(e_2)$ are still Gaussian; the integral
reduces to a Gaussian-Gaussian convolution with effective $\sigma_M^2 =
\sigma^2\ell_M^2/(\sigma^2 + \ell_M^2)$:

$$
I_{M,K,p}
 = \tau^2 \cdot \left(\frac{\ell_M^2}{\sigma^2 + \ell_M^2}\right)^{2}
 \cdot \frac{\ell^2}{\ell^2 + 2\sigma_M^2}.
$$

For **eccentric** and **linear** masks the integrands are
Gaussian-times-bounded-functions; the closed forms exist but are tedious,
and we use Monte Carlo (4M-sample default) via
`synthetic.ground_truth(...)`, with sampling noise $\lesssim 10^{-3}$ —
well below the test tolerances.

The ratio $1-\alpha^{D,w} = 1 - I_{M,K,D}/(\tau^2 \mathbb E_D[M^2])$ is
invariant under time-bin weighting and the envelope $\alpha(t)$: both
$\mathrm{Var}_\text{PSTH}$ and $\mathrm{Var}_\text{total}$ are proportional
to $\mathbb E_w[\alpha^2]$, which cancels. Both Extension-1 directions
therefore target the same closed-form ratio; the choice between them is a
finite-sample efficiency question, not a bias question.

## A.6 Consistency: how $\mathrm{sd}[1-\hat\alpha]$ depends on $N$ and $T$, and the $[0,1]$ clipping bias

Section 2.4 noted that the seed-to-seed SEM of McFarland's estimator does
**not** shrink as $1/\sqrt{N}$ alone — an across-time-bin noise floor in $T$
kicks in once within-bin sampling is adequate, and at high SEM the
$[0,1]$ clipping of $\hat\alpha$ introduces a mean bias. This appendix
derives both effects on the unified flat-mask synthetic under (A1)+(A2)
and calibrates them empirically against the closed form. The code is
`fig_consistency.py`; the empirical sweep is cached to
`consistency_sweep.npz`.

### A.6.1 The across-time-bin floor

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
 \;=\; \frac{\tau^2\,\ell^2}{\ell^2+2\sigma^2}
 \;=\; \alpha^*\,\tau^2,
\qquad
\alpha^* \equiv \frac{\ell^2}{\ell^2+2\sigma^2}.
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

a finite limit in $T$ alone. At $\ell=\sigma$ (so $\alpha^*=1/3$) and
$T=100$, (A6.1) gives $\mathrm{sd}\approx 0.0474$, in agreement with the
leveling-off observed in §2.4.

### A.6.2 Within-bin contribution and the empirical decomposition

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
over $(N,T)$ at $\ell=\sigma$: the right edge (large $T$, $\mathrm{sd}\approx
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
real-data numbers in §5.2 — those are robust to it at the population
median (Δ $1-\alpha = -0.022$) — but it matters for per-cell SEM and is
the operative reason the within-stimulus-frame reliability question is
flagged as a future direction (§2.3).

### A.6.3 Boundary-clipping bias

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
(A6.3) on the small-$\ell$ side: at $\ell=0.3\sigma$ ($\alpha^*=0.043$),
each $(N,T)$ cell produces an $(\mathrm{sd},\mathrm{bias})$ pair, and the
empirical points sit on the analytical curve as $\sigma_\alpha$ is varied.
The bias reaches $\sim -0.08$ at $\sigma_\alpha\approx 0.3$ — small in
absolute terms, but a $9\%$ shift relative to the true $1-\alpha^*=0.957$.
The same effect explains the saturation of $1-\hat\alpha$ at $0$ for
eccentric-mask cells in §5.1, where the unclipped close-pair $\hat\alpha$
excursions above $1$ from the inflated-$C_\text{rate}$ side of the naive
inconsistency.

![**Figure A6 — SEM and $[0,1]$ clipping of $1-\hat\alpha$ on the
flat-mask synthetic.** **(A)** Empirical seed-to-seed $\mathrm{sd}[1-\hat\alpha]$
over a 4×4 $(N,T)$ grid at $\ell=\sigma$ (10 seeds per cell, deterministic
rates, `target='naive'`, threshold 0.05). The T-floor
$\alpha^*\sqrt{2/(T-1)}$ is printed below the panel; empirical SD shrinks
toward it as $N$ grows and is bounded below by it as $T$ shrinks. **(B)**
Boundary clipping at $\ell=0.3\sigma$ ($\alpha^*=0.043$,
$1-\alpha^*=0.957$): bias of $1-\hat\alpha$ from truth, one marker per
$(N,T)$ cell, coloured by $T$. The dashed curve is the analytical
truncated-Gaussian prediction (A6.3). **(C)** $\mathrm{sd}[1-\hat\alpha]$
vs $T$ at $\ell=\sigma$ for $N\in\{100,200,400,800\}$. The dashed line is
the closed-form floor $\alpha^*\sqrt{2/(T-1)}$; the large-$N$ points sit
on it.](figures/fig_consistency.png)

## A.7 The PSTH covariance estimator: McFarland M6 vs bagged split-half

The §1 estimators of $C_{\text{psth}}$ and $C_{\text{rate}}$ both rest on the
cross-trial trick of (4) — distinct-trial products at the same time bin have
independent observation noise, so the average product converges to the
underlying rate product with the noise removed. The close-pair estimator (8) is
one specific arrangement of this trick (pairs restricted to $\Delta e<\varepsilon$).
The PSTH-covariance estimator (6) is another (pairs unrestricted on $\Delta e$).
Two computationally distinct estimators target (6); this appendix walks through
why the naive PSTH covariance fails, names the two distinct-trial estimators
that fix it, and records the choice this writeup makes.

### A.7.1 Why the naive PSTH covariance is biased — and worse on off-diagonals

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

### A.7.2 Two distinct-trial estimators of $C_{\text{psth}}$

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

### A.7.3 Tradeoffs

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

### A.7.4 Our choice — McFarland M6

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
uv run python fig_sanity_check.py
uv run python fig_time_bin_weighting.py
uv run python fig_naive_failure.py
uv run python fig_correction.py
uv run python fig_consistency.py            # parallel sweep; cached to .npz
uv run python generate_realdata.py          # cache-only; --recompute to rebuild
uv run --with pytest pytest test_estimators.py -q
```

*Build this note to a self-contained, offline HTML (math via MathML, images
inlined):*

```bash
pandoc writeup.md -s --mathml --self-contained -o writeup.html
```
