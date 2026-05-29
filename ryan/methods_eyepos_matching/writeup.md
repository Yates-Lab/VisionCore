---
title: "Eye-position-distribution matching in the cross-trial decomposition of V1 responses under fixational eye movements"
subtitle: "A methodological advance for non-homogeneous stimuli"
author: "fem-v1-fovea methods note"
date: "2026-05-29"
---

# Summary

The cross-trial, eye-conditioned decomposition of McFarland, Cumming & Butts (2016)
is the standard tool for separating the stimulus-driven and stimulus-independent
components of V1 responses in the presence of fixational eye movements (FEMs). It
estimates the stimulus-driven *rate* variance by conditioning on pairs of distinct
trials whose eye positions are nearly identical — a device that cancels the
independent Poisson observation noise. We show that this close-pair conditioning
silently fixes the *eye-position distribution* over which the rate variance is
measured: close pairs are sampled in proportion to the **squared** eye-position
density $p(e)^2$, which is concentrated near the center of fixation, whereas the
total covariance and the PSTH covariance are measured over the full density $p(e)$.

For a **homogeneous** stimulus — one whose statistics are invariant to spatial
translation, as in McFarland et al.'s ternary bar noise — this mismatch is
harmless, and McFarland et al. note as much. For a **non-homogeneous** stimulus —
such as the structured `fixRSVP` images used here, where the retinal image shifts
with drift so that the firing rate depends on absolute eye position — the mismatch
biases the law-of-total-covariance (LOTC) decomposition: it corrupts the FEM
fraction $1-\alpha$, the FEM-corrected noise correlation $C_{\text{noise}}^{C}$, and
the Fano factor, with an error whose **sign is set by the spatial profile of each
cell's eye-position sensitivity** (central vs. eccentric).

We make the dependence explicit, show with synthetic ground truth that the naive
(distribution-unmatched) estimator fails in exactly the predicted way, and develop
a corrected estimator that matches the eye-position distribution across all terms.
The target distribution is a free parameter with two principled choices — the full
fixational distribution $p(e)$ (the actual viewing conditions; the scientifically
natural target) and the central distribution $p(e)^2$ (statistically more stable) —
which **coincide if and only if the stimulus is homogeneous**. Their difference is
therefore a direct, quantitative measure of stimulus non-homogeneity.

---

# 1. Background: the cross-trial decomposition under FEMs

## 1.1 The law of total (co)variance

Let $Y_c^i(t)$ be the spike count of neuron $c$ on trial $i$ in the counting window
at stimulus phase $t$ (a frozen-stimulus time bin). Each trial samples an absolute
eye position $e$ from the fixational distribution $p(e)$; we treat $e$ as
approximately constant over the short counting window and approximately independent
of $t$. Writing the firing rate as $r_c(t,e) = \mathbb{E}[Y_c \mid t, e]$, the
**law of total variance** partitions the stimulus-driven rate variance into a
stimulus-locked (PSTH) part and an eye-movement (FEM) part:

$$
\underbrace{\mathrm{Var}_{t,\,e\sim p}\!\big[r_c(t,e)\big]}_{\text{total rate variance}}
 \;=\;
\underbrace{\mathrm{Var}_t\!\Big[\,\mathbb{E}_{e\sim p}\!\big[r_c \mid t\big]\Big]}_{\text{PSTH variance}}
 \;+\;
\underbrace{\mathbb{E}_t\!\Big[\,\mathrm{Var}_{e\sim p}\!\big[r_c \mid t\big]\Big]}_{\text{FEM variance}} .
\tag{1}
$$

The PSTH at phase $t$ is the across-trial mean rate $\mathbb{E}_{e\sim p}[r_c\mid t]$,
i.e. an average over the **full** eye-position distribution. McFarland et al. define
the fraction of stimulus-driven variance captured by the PSTH,

$$
\alpha_c \;=\; \frac{\mathrm{Var}_t\big[\mathbb{E}_{e\sim p}[r_c\mid t]\big]}
                     {\mathrm{Var}_{t,e\sim p}[r_c]} ,
\qquad
1-\alpha_c \;=\; \text{FEM fraction}.
\tag{2}
$$

The pairwise analogue partitions the count covariance:
$\mathrm{Cov}_{t,e\sim p}[Y_m,Y_n] = \mathrm{Cov}_{t,e\sim p}[r_m,r_n] + \mathbb{E}\big[\text{noise cov}\big]$,
so the FEM-corrected (stimulus-independent) covariance is

$$
C_{\text{noise}}^{C} \;=\; C_{\text{total}} \;-\; C_{\text{rate}},
\qquad
\text{Fano}_c \;=\; \frac{\big(C_{\text{noise}}^{C}\big)_{cc}}{\bar r_c},
\tag{3}
$$

where $C_{\text{total}} = \mathrm{Cov}_{t,e\sim p}[Y]$ is the raw count covariance and
$C_{\text{rate}} = \mathrm{Cov}_{t,e\sim p}[r]$ is the **total rate covariance over the
same distribution $p$**. Equation (3) is only meaningful if $C_{\text{rate}}$ and
$C_{\text{total}}$ are measured over one and the same eye-position distribution; this
is the consistency requirement that the rest of this note is about.

## 1.2 The cross-trial trick: distinct trials cancel the observation noise

We never observe $r_c(t,e)$ directly — only the noisy counts $Y_c$. The engine of the
whole method is that the observation noise is independent across **distinct** trials,
so for two different trials $i\neq j$ at the same phase $t$,

$$
\mathbb{E}\big[Y_c^i(t)\,Y_c^j(t)\,\big|\,e_i,e_j\big] = r_c(t,e_i)\,r_c(t,e_j).
\tag{4}
$$

The product of two distinct trials' counts is an unbiased estimate of the product of
their underlying rates, with the Poisson/observation variance removed (that variance
would have contaminated a single trial's squared count). Every estimator below is a
different way of averaging these distinct-trial products — and **the choice of which
pairs to average sets the eye-position distribution**, which is the whole story.

## 1.3 Two families of estimator: the signal (PSTH) and the eye-conditioned rate

**Signal (PSTH) variance and covariance — all distinct pairs.** Averaging (4) over
*all* distinct pairs at the same phase, then over phases, estimates the PSTH
(stimulus-locked) second moment. McFarland et al.'s PSTH-variance estimator (their
Eq. 6) and its multineuron analogue (the "shuffle corrector") are

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

Because *all* pairs are used, the phase mean averages over the **full** fixational
distribution $p(e)$: these terms live on $p(e)$.

**Eye-conditioned rate variance and covariance — close pairs only.** Restricting the
same averages to pairs whose eye trajectories nearly coincide, $\Delta e_{ij}<
\varepsilon$, drives $e_i\approx e_j\approx e$, so (4) $\to r_c(t,e)^2$: the *total*
rate second moment with the Poisson noise removed (their Eqs. 8 and 16),

$$
\widehat{\mathrm{Var}_{i,t}[r_c]}
 = \big\langle \langle Y_c^i(t)\,Y_c^j(t)\mid \Delta e_{ij}<\varepsilon\rangle_{i\ne j}\big\rangle_t - \bar Y_c^{\,2},
\tag{7}
$$
$$
\widehat{\mathrm{Cov}_{i,t}[r_m,r_n]}
 = \big\langle \langle Y_m^i(t)\,Y_n^j(t)\mid \Delta e_{ij}<\varepsilon\rangle_{i\ne j}\big\rangle_t - \bar Y_m\,\bar Y_n,
\tag{8}
$$

where the eye-trajectory distance is an RMS over a short window of $T$ samples (their
Eq. M10),

$$
\Delta e_{ij}(t) = \Big(\tfrac{1}{T}\textstyle\sum_{\tau}
   \big\lVert e^i(t-\tau)-e^j(t-\tau)\big\rVert^2\Big)^{1/2}.
\tag{9}
$$

The close-pair restriction is exactly what cancels the Poisson noise across distinct
trials — but, as §2 shows, it also silently changes the eye-position distribution
these terms live on.

## 1.4 The reported quantities, and how they combine the two families

**FEM fraction.** $\alpha$ is the share of stimulus-driven variance captured by the
PSTH (their Eq. 4); $1-\alpha$ is the FEM fraction:

$$
\alpha_c \;=\; \frac{\widehat{\mathrm{Var}_t[\mathbb E_i r^i_c]}}
                    {\widehat{\mathrm{Var}_{i,t}[r_c]}}
        \;=\; \frac{(5)}{(7)}.
\tag{10}
$$

McFarland et al. also give an analytical form (their Eq. 9). With gaze position $x$,
the PSTH is the rate **convolved with the eye-position distribution**, $\bar r = r*p$,
so by Parseval

$$
\alpha = \frac{\int |R(k)|^2\,|P(k)|^2\,dk}{\int |R(k)|^2\,dk},
\qquad R=\mathcal F[\bar r],\; P=\mathcal F[p].
\tag{11}
$$

This makes the role of $p$ explicit: $\alpha$ weights the rate spectrum $|R(k)|^2$ by
the eye-distribution spectrum $|P(k)|^2$ (a low-pass set by the fixation spread).

**Noise covariance, noise correlation, and Fano factor.** The stimulus-independent
("noise") covariance is the total count covariance minus the rate covariance (their
Eq. 14); the Fano factor and its FEM-induced bias follow (their Eqs. 10, 13):

$$
C_{\text{noise}}^{C}
 = \underbrace{\mathrm{Cov}_{i,t}[Y_m,Y_n]}_{C_{\text{total}}}
 - \underbrace{\widehat{\mathrm{Cov}_{i,t}[r_m,r_n]}}_{C_{\text{rate}}\ \text{from (8)}},
\qquad
\text{FF bias} = (1-\alpha)\,\frac{\mathrm{Var}_{i,t}[r]}{\bar Y}.
\tag{12}
$$

The noise correlation is $C_{\text{noise}}^{C}$ normalized to unit diagonal.

**How our pipeline computes these** (`VisionCore/covariance.py`):

- **$C_{\text{total}}$** — `torch.cov` over **all** counting windows → full $p(e)$.
- **$C_{\text{psth}}$** — bagged split-half of the per-phase means
  (`bagged_split_half_psth_covariance`), the (5)/(6) family → full $p(e)$.
- **$C_{\text{rate}}$** — the close-pair `below_threshold` intercept
  (`estimate_rate_covariance`, $\Delta e<0.05^\circ$), the (7)/(8) family → see §2.

---

# 2. The problem: a non-homogeneous stimulus pins $C_{\text{rate}}$ to a different distribution

## 2.1 The homogeneity assumption, stated explicitly

The eye-conditioned estimators (7)–(8) restrict the average to close pairs. McFarland
et al. flag the consequence directly (their text around Eqs. M7–M10):

> "by restricting analysis to trial pairs where $\Delta e_{ij}\approx 0$, [the
> estimator] gives an estimate of $\mathbb{E}[r^2(e,t)]$ under the conditional eye
> position distribution $p(e\mid\Delta e\approx 0)$ rather than $p(e)$. However, for
> a stimulus that is statistically invariant to spatial translations (such as used
> in our study), the expectation of $r^2(e,t)$ with respect to any distribution over
> $e$ will be the same … Thus, in this case, sampling $r^2(e,t)$ across eye positions
> at a given time point is equivalent, on average, to sampling … at a given eye
> position."

This is the **homogeneous-stimulus assumption**: under spatial-translation
invariance, the eye-position distribution over which the rate variance is measured
does not matter, so the close-pair restriction in (7)–(8) is free. Their ternary bar
noise satisfies it. Our `fixRSVP` stimulus does not — for two distinct reasons.

## 2.2 Why `fixRSVP` is non-homogeneous: a windowed image

`fixRSVP` presents a natural image at a fixed location on the screen, behind a
spatial **window** (aperture). The neuron's receptive field (RF) is fixed in
retinotopic coordinates, so in screen coordinates it moves with the eye. Two
mechanisms make $r_c(t,e)$ depend on absolute eye position $e$, in direct violation of
translation invariance:

1. **Windowing — a hard, trivial non-homogeneity.** Once the fixation offset is large
   enough to carry the RF off the windowed image, the RF samples only the uniform gray
   background, and the rate collapses to the gray-screen baseline **regardless of
   stimulus phase $t$** — the stimulus-driven modulation simply vanishes. So
   $r(t,e)$ is structurally eye-position-dependent — strongly modulated near the
   center of fixation, flat at baseline in the periphery — *before any image content
   is considered*. This alone breaks the assumption.
2. **Image structure.** Within the window, drift slides the structured image across
   the RF, so which feature drives the cell, and how strongly, depends on absolute eye
   position — the ordinary translation-variance of any non-noise stimulus.

The windowing case is especially clarifying. The close-pair (central) distribution
$p(e)^2$ over-represents the eye positions where the image is *on* the RF and the cell
is strongly driven; the full distribution $p(e)$ also includes peripheral positions
where the RF sees gray and the cell is at baseline. The PSTH (over $p$) averages in
those near-baseline peripheral samples, while the close-pair rate variance (over
$p^2$) does not — so the two are measured on **materially different stimulus drive**,
and $1-\alpha$, the noise covariance, and the Fano factor all inherit the mismatch.

## 2.3 Close pairs are sampled from the squared density $p(e)^2$

Take two trials with eye positions $e_i, e_j$ drawn independently from $p$. The
density of close pairs *at position $e$* — pairs with $e_i\approx e_j\approx e$ within
the threshold ball — is proportional to $p(e)\cdot p(e) = p(e)^2$. Hence the close-pair
estimator (7) does not measure $\mathbb{E}_{e\sim p}[r^2]$ but

$$
\big\langle Y_c^i Y_c^j \mid \Delta e<\varepsilon\big\rangle
 \;\;\longrightarrow\;\;
\mathbb{E}_{e\sim p^2}\!\big[r_c(t,e)^2\big]
\quad (\varepsilon\to0),
\qquad
p^2(e) \equiv \frac{p(e)^2}{\int p(e')^2\,de'} .
\tag{13}
$$

For an isotropic Gaussian fixation $p=\mathcal N(0,\sigma^2 I)$ the close-pair
distribution is *exactly* $p^2=\mathcal N\!\big(0,\tfrac{\sigma^2}{2}I\big)$: a tighter,
more central Gaussian with **half the variance**. Figure 1 confirms this
geometrically and numerically.

![**Figure 1 — Close pairs sample the squared density $p(e)^2$.** **(A)** Eye
positions (grey) and the representative positions of distinct-trial close pairs
(red) for $\sigma=0.15^\circ$, threshold $0.05^\circ$; the $1,2\sigma$ circles of $p$
(solid) enclose the tighter $1,2\sigma/\sqrt2$ circles of $p^2$ (dashed). **(B)** The
$x$-marginal: the close-pair distribution matches $\mathcal N(0,\sigma^2/2)$ (observed
variance ratio $\approx 0.5$). **(C)** Consequently the FEM variance $\mathrm{Var}[F]$
of an eye-sensitivity profile differs between $p$ and $p^2$, and the sign of the
resulting $1-\alpha$ bias depends on the profile.](figures/fig_mechanism.png)

## 2.4 The decomposition is consistent only on one distribution

Combining §1 and §2.3: in the non-homogeneous case the pipeline measures

$$
C_{\text{total}},\,C_{\text{psth}} \ \text{over } p(e),
\qquad
C_{\text{rate}} \ \text{over } p(e)^2 ,
$$

so the derived quantities mix two distributions,

$$
1-\alpha = 1 - \frac{C_{\text{psth}}(p)}{C_{\text{rate}}(p^2)},
\qquad
C_{\text{noise}}^{C} = C_{\text{total}}(p) - C_{\text{rate}}(p^2).
\tag{14}
$$

The law of total covariance (1) holds term-by-term **only when every term is taken
over the same distribution**. Equation (14) violates this whenever
$\mathbb{E}_{p^2}[r^2]\neq\mathbb{E}_p[r^2]$, i.e. whenever the rate depends on
absolute eye position — precisely the non-homogeneous case.

## 2.5 A sharper inconsistency: the mean and the second moment disagree

There is a second, subtler defect already present in the *current* estimator. The
close-pair second moment in (7) and (13) is taken over $p^2$, but the term subtracted
from it — $\bar Y_c^2$, the global mean — is taken over $p$
(`estimate_rate_covariance` subtracts a full-distribution mean, `covariance.py:820,856`):

$$
\big(C_{\text{rate}}\big)_{cc}^{\text{naive}}
 = \underbrace{\mathbb{E}_{p^2}[r_c^2]}_{\text{2nd moment over }p^2}
 - \underbrace{\big(\mathbb{E}_{p}[r_c]\big)^2}_{\text{mean over }p}.
\tag{15}
$$

This is **not a variance under any single distribution**. Because
$\mathbb{E}_{p^2}[r]>\mathbb{E}_p[r]$ for a centrally-peaked profile and
$\mathbb{E}_{p^2}[r]<\mathbb{E}_p[r]$ for an eccentric one, the naive $C_{\text{rate}}$
can be *inflated above* the true $p^2$ variance (central cells) or driven *negative*
(eccentric cells). We confirm both below. The prior phase-weighting fix (matching
the $n_t^2$ time-bin weighting of $C_{\text{psth}}$ and $C_{\text{rate}}$, validated by
a shuffle null) is a **different axis**: it corrects how phases are weighted, not the
within-phase eye-position distribution. Crucially, **the shuffle null is blind to the
present problem**: shuffling eye trajectories across trials destroys the eye–spike
coupling, removing exactly the $r(e)$ dependence on which the distribution mismatch
acts, so $D_z\approx0$ under shuffling says nothing about eye-position-distribution
consistency.

---

# 3. How the naive estimator fails (synthetic ground truth)

We generate synthetic V1 cells with a known rate field
$r_c(t,e) = \text{base} + a\,\bar P_c(t) + b\,F_c(e)$ — a zero-mean phase-locked drive
$\bar P_c(t)$ and an eye-sensitivity profile $F_c(e)$ — and draw Poisson spikes at
real-istic fixation $\sigma=0.15^\circ$ (module `synthetic.py`). Because $p^2$ for a
Gaussian is again Gaussian, the LOTC decomposition (1) has a closed form under both
$p$ and $p^2$, obtained by direct sampling; we verified that the trusted all-samples
ANOVA recovers it when fed eyes from the matching distribution. Four profiles span
the relevant cases: `flat` (homogeneous), `central` (peaked at fixation), `eccentric`
($\propto\lvert e\rvert^2$), and `linear`.

Figure 2 shows the failure on pure-Poisson data, where the true noise correlation is
$0$ and the true Fano factor is $1$:

- **$1-\alpha$** (panel A): the naive estimator **over-states** the FEM fraction for
  central cells (sitting far above the identity line) and **under-states** it — to the
  point of an undefined, negative $C_{\text{rate}}$ (NaN) — for eccentric cells. The
  matched estimator lands on the truth.
- **Noise correlation** (panel B): the rate-variance distribution mismatch leaks into
  $C_{\text{total}}-C_{\text{rate}}$, producing **spurious** stimulus-independent
  correlations (median $\lvert r\rvert$ far from $0$) where the truth is exactly $0$.
- **Fano factor** (panel C): the same leak biases the Fano factor away from $1$, with
  a population that splits off toward inflated values.

![**Figure 2 — The naive (distribution-unmatched) estimator fails on all three
reported quantities**, with a sign set by each cell's eye-sensitivity profile. Pure
Poisson synthetic data (true noise correlation $0$, true Fano $1$); "matched" is the
$p$-target corrected estimator. **(A)** $1-\alpha$ vs. ground truth: naive (×) biased,
matched (○) on the identity line. **(B)** spurious noise correlation. **(C)** biased
Fano factor.](figures/fig_naive_failure.png)

The sign rule echoes the Figure-4 mechanism result
(`ryan/fig4/mechanism_fig4d.py`): conditioning on $\Delta e<\varepsilon$ weights the
FEM by $p(e)^2$, so a profile whose sensitivity is **central** is *over*-weighted
($p^2$ emphasizes the center) and one whose sensitivity is **eccentric** is
*under*-weighted.

---

# 4. The corrected estimator: matching the eye-position distribution

## 4.1 A single device with the target distribution as a parameter

The close-pair conditioning is *required* to cancel the independent Poisson noise, so
we cannot avoid sampling from $p^2$. But the close pairs give Poisson-free **local**
estimates of $r(t,e)^2$; the bias lives entirely in how those local estimates are
**aggregated over $e$**. We therefore re-weight toward a chosen target distribution
$q(e)$ by importance sampling. A sample drawn from $p$ contributes weight
$q(e)/p(e)$; a close *pair* drawn from $p^2$ contributes weight $q(e)/p(e)^2$:

$$
\widehat{\mathbb{E}_q[g]}
 = \frac{\sum_{\text{pairs}} \tfrac{q(e)}{p(e)^2}\, g}{\sum_{\text{pairs}} \tfrac{q(e)}{p(e)^2}}
 \quad\text{(second moment)},
\qquad
\widehat{\mathbb{E}_q[h]}
 = \frac{\sum_{\text{samples}} \tfrac{q(e)}{p(e)}\, h}{\sum_{\text{samples}} \tfrac{q(e)}{p(e)}}
 \quad\text{(mean, total, PSTH)} .
\tag{16}
$$

Two choices of $q$ make all terms consistent (module `estimators.py`, `target=`):

| target | $q$ | close-pair weight $q/p^2$ | sample weight $q/p$ | character |
|---|---|---|---|---|
| **Direction 1** (`full`) | $p$ | $1/p$ (unbounded) | $1$ | actual viewing distribution |
| **Direction 2** (`central`) | $p^2$ | $1$ | $\propto p$ (bounded) | central; close-pair-supported |
| (naive) | — | $1$ on $p^2$ | $1$ (mean over $p$) | inconsistent (Eq. 15) |

Both directions fix the mean/second-moment inconsistency of (15) by construction
(the mean is taken over the same $q$ as the second moment). They are **the two
consistent resolutions** of the mismatch: push everything out to the full
distribution, or pull everything in to the close-pair distribution.

## 4.2 Equivalence to eye-position stratification

Importance reweighting (16) is equivalent to **stratifying by absolute eye position**:
partition $e$ into strata, estimate the close-pair second moment *within* each
stratum (where $p\approx$ const so $p^2\approx p$ locally and the local estimate is
unbiased for $\mathbb E[r^2\mid s]$), then aggregate strata with weights equal to the
target occupancy $q(s)$. As the strata shrink this is exactly (16); with finite strata
it pools pairs before weighting, trading a little resolution for lower variance. This
is the natural generalization of McFarland et al.: their estimator conditions on
phase $t$ and on $\Delta e\approx0$; the correction conditions *additionally* on
absolute eye position.

## 4.3 Recovery, and the Direction-1 vs. Direction-2 tradeoff

Figure 3A shows that each target recovers its own closed-form decomposition (points
on the identity line), up to a small finite-threshold smoothing that shrinks as the
threshold shrinks (a limitation shared with the original McFarland estimator: a
spatial feature narrower than $\varepsilon$ is smoothed).

The two directions are **not** equally easy to estimate (Figure 3B). Direction 1
(target $p$) requires the unbounded weight $1/p$, which is largest in the periphery —
exactly where close pairs are rarest. For an eccentric-sensitive cell, whose variance
*lives* in the periphery, this makes the estimate noisy: its across-seed standard
deviation grows as the threshold shrinks. Direction 2 (target $p^2$) uses bounded
weights $\propto p$, largest at the center where close pairs are abundant, and is
markedly more stable.

This is the crux of the design choice:

- **Direction 1 ($p$) is the scientifically natural target.** The Fano factor, noise
  correlation, and FEM fraction are properties of the neuron *under the actual viewing
  conditions*, whose eye-position distribution is $p$. A quantity reported over $p^2$
  is "as if the animal fixated more tightly than it did," which would require a caveat
  on every number. We therefore headline Direction 1 for the paper's reported values.
- **Direction 2 ($p^2$) is the stable cross-check.** Where Direction 1 is too noisy
  (eccentric cells, small thresholds, little data), Direction 2's bounded weights give
  a reliable, if differently-targeted, estimate.

Finally (Figure 3C), the **gap** $\lvert(1-\alpha)_{\text{full}} -
(1-\alpha)_{\text{central}}\rvert$ is itself informative: it is $\approx0$ for a
homogeneous (`flat`) cell and grows with the spatial structure of the
eye-sensitivity profile. Since the two consistent targets coincide *iff* the stimulus
is homogeneous, the gap is a direct, model-free **measure of stimulus
non-homogeneity** — an empirical handle on exactly the assumption McFarland et al.
make.

![**Figure 3 — The matched estimator recovers ground truth and exposes the
Direction-1/Direction-2 tradeoff.** **(A)** `full` recovers the $p$ decomposition and
`central` the $p^2$ decomposition (identity line). **(B)** For an eccentric cell,
Direction 1's unbounded $1/p$ weights make $1-\alpha$ noisy as the threshold shrinks;
Direction 2 is stable. **(C)** The full-vs-central gap is $\approx0$ for a homogeneous
cell and grows with non-homogeneity — a model-free diagnostic.](figures/fig_correction.png)

---

# 5. Consequences on real data

We applied the estimator to the real `fixRSVP` recordings, cache-only (no GPU, no
model inference; `generate_realdata.py` reads the Figure-4 cache of trial-aligned
spikes and real eye trajectories). $1-\alpha$ and the Fano factor are computed on the
real spikes with each cell's own validity mask (which reproduces the Figure-2 per-cell
$1-\alpha$ at the median); the eye-position density is a Gaussian KDE of the measured
fixational positions. Pooled over **397 good cells** ($\mathrm{cc}_{\max}>0.85$, 2
monkeys, 24 sessions):

| quantity | naive | Direction 1 ($p$, `full`) | Direction 2 ($p^2$, `central`) |
|---|---|---|---|
| median $1-\alpha$ | **0.734** | 0.702 | 0.608 |
| median Fano | 0.846 | 0.875 | — |

- **The naive bias on population $1-\alpha$ is small.** The naive median (0.734)
  reproduces the Figure-2 value (0.732) and lies only $+0.022$ above the
  Direction-1-corrected value (0.702). On the actual-viewing target $p$, the existing
  Figure-2 / Figure-4 panel-D $1-\alpha$ conclusions are therefore **robust** to the
  distribution mismatch at the population level — real V1 cells do not behave like the
  extreme synthetic `central`/`eccentric` profiles, which suffer large biases.
- **But there is measurable non-homogeneity.** The gap between the two consistent
  targets, $\lvert(1-\alpha)_{\text{full}}-(1-\alpha)_{\text{central}}\rvert$, has a
  population **median of 0.089** with a tail beyond 0.3 (Figure 4B). Since the two
  targets would coincide for a homogeneous stimulus, this gap is direct evidence that
  the `fixRSVP` stimulus is non-homogeneous for a substantial fraction of cells, and
  it sets the scale at which the *choice* of eye-position distribution matters for
  $1-\alpha$.
- **The Fano factor shifts modestly** under matching (median $0.846\to0.875$, $+3\%$),
  consistent with the synthetic prediction that the Fano factor inherits the
  rate-variance distribution mismatch (Figure 2C); per-cell shifts are larger.

![**Figure 4 — The correction on real data (397 good cells, cache-only).**
**(A)** $1-\alpha$ on real spikes: Direction 1 (blue) tracks the naive estimate
closely (median shift $-0.022$), while Direction 2 (red) is systematically lower.
**(B)** The full-vs-central gap — a model-free non-homogeneity measure — has median
0.089 with a heavy tail. **(C)** Fano factor: naive vs. matched, a modest median
shift with larger per-cell changes.](figures/fig_realdata.png)

The **noise-correlation** consequence is established on synthetic ground truth
(Figure 2B), where the truth is exactly zero and the naive estimator produces large
spurious correlations that the matched estimator removes. Quantifying it on real
*spike* pairs requires either the full windowed pipeline or careful joint-pair
validity masking; because it shares the same pipeline change, we fold it into the
gated Figure-2 fix (§6) rather than approximate it here. (Computing it on the
deterministic model rates is not informative: with no observation noise the true
$C_{\text{noise}}^{C}$ is $\approx0$, so its *correlation* is an ill-posed
$0/0$.)

---

# 6. Fixing Figure 2

The covariance machinery and caches are shared by Figures 2–4, so a pipeline change
has broad blast radius. The corrected estimator is implemented in this folder
(`estimators.py`) and validated by the TDD suite (`test_estimators.py`). The proposed
production change is to add the target-distribution importance weights to
`estimate_rate_covariance` (close-pair weight $q/p^2$) and to the per-sample weighting
of `torch.cov`/`bagged_split_half_psth_covariance` (sample weight $q/p$) in
`VisionCore/covariance.py`, gated behind a `target` argument that defaults to the
current behavior, with `target='full'` reproducing the headline numbers over $p(e)$.

The regeneration of the Figure-2 decomposition cache requires a GPU and is expensive;
**it is gated on explicit approval** and is not performed by this note. The expected
direction of the change is set by §5.

---

# Appendix: derivations

## A.1 Close-pair density is $p(e)^2$

Let $e_i,e_j\stackrel{\text{iid}}{\sim}p$. The probability that a pair is "close" with
both members in a small ball $B_\delta(e)$ around $e$ is
$\Pr[e_i\in B_\delta(e)]\,\Pr[e_j\in B_\delta(e)] \approx \big(p(e)\,|B_\delta|\big)^2$.
Dividing by the total close-pair mass and taking $\delta\to0$ gives the close-pair
position density $p(e)^2/\int p^2$. For $p=\mathcal N(0,\sigma^2 I)$ in $d$ dimensions,
$p(e)^2\propto\exp(-\lVert e\rVert^2/\sigma^2)=\mathcal N(0,\tfrac{\sigma^2}{2}I)$ up to
normalization — variance halved.

## A.2 Importance weights

To estimate $\mathbb E_q[g] = \int g\,q$ from samples distributed as $s$, write
$\mathbb E_q[g] = \int g\,\tfrac{q}{s}\,s = \mathbb E_s\!\big[g\,\tfrac{q}{s}\big]$, so
each sample is weighted by $q/s$. For the second moment the close pairs have $s=p^2$,
giving weight $q/p^2$ (Direction 1: $p/p^2=1/p$; Direction 2: $p^2/p^2=1$). For the
mean/total/PSTH the trials have $s=p$, giving weight $q/p$ (Direction 1: $1$;
Direction 2: $p$). Substituting into the empirical LOTC terms yields a decomposition
in which all of $C_{\text{total}}$, $C_{\text{psth}}$ and $C_{\text{rate}}$ are taken
over the single distribution $q$, restoring term-by-term consistency of (1).

## A.3 When the correction is a no-op

If $r_c(t,e)$ does not depend on $e$ (homogeneous), then $\mathbb E_q[r^2]=r^2$ and
$\mathbb E_q[r]=r$ for every $q$, so all targets coincide and the importance weights
cancel: the correction reduces to the identity, as required, and as confirmed by the
`flat`-profile test.

---

*Reproduce all figures (from this folder):*

```bash
uv run python fig_mechanism.py
uv run python fig_naive_failure.py
uv run python fig_correction.py
uv run python generate_realdata.py          # cache-only; --recompute to rebuild
uv run --with pytest pytest test_estimators.py -q
```

*Build this note to a self-contained, offline HTML (math via MathML, images inlined):*

```bash
pandoc writeup.md -s --mathml --self-contained -o writeup.html
```
