---
title: "Directly estimating the close-pair density vs the squared-marginal assumption"
subtitle: "An implementation note to `writeup.md` (methods_eyepos_matching)"
author: "fem-v1-fovea methods note"
date: "2026-06-24"
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

The eye-position-matched estimator of the main writeup (§4.2, §4.4) reweights
every term toward a target eye distribution by importance sampling. The
close-pair second moment is sampled from the **close-pair density** $p_{\mathrm{pair}}$, and the importance weights need that density. Throughout, the writeup
substitutes the squared marginal $\hat p^2$ for $p_{\mathrm{pair}}$ — the §A.5
result that close pairs drawn from $p$ have midpoint density $\propto p(e)^2$,
*exact in the single-bin $\Delta e\to 0$ limit*. In the production trajectory-mode
estimator (§4.4) each sample's eye trajectory is reduced to a geometric-median
representative point $\rho_i$ and "close" is judged by the whole-window RMS
distance (17), so that identity is only approximate: neither the finite
threshold nor the trajectory reduction is guaranteed to leave $p_{\mathrm{pair}} =
\hat p^2$.

This note tests the assumption directly. We add a `closepair_density={'direct'
(default),'squared'}` switch to the estimator (`estimators.decompose`,
`decompose_trajectory`, threaded through `pipeline.decompose_session`).
`'direct'` fits a second KDE $\hat p_{\mathrm{pair}}$ on the realized close-pair
representative-midpoints instead of squaring $\hat p$. On the same 25 real
`fixRSVP` sessions and the exact §4.5 estimator window we find:

- **The squared marginal is measurably wrong, in a consistent direction.**
  The close-pair variance ratio $\operatorname{tr}\operatorname{cov}(\rho_{\mathrm{mid}})/\operatorname{tr}\operatorname{cov}(\rho)$ has population median
  $0.42$ — *below* the ideal Gaussian value $0.5$ — with
  $\mathrm{KL}(\hat p_{\mathrm{pair}}\,\|\,\hat p^2)$ median $0.16$. Real close pairs
  are **more central** than $p^2$ predicts, because real fixation densities are
  more peaked than Gaussian and the RMS-trajectory match preferentially keeps
  the stablest central fixations.

- **The headline result is robust.** On the reported Direction 1 (`full`,
  target $p$), the population median $1-\alpha$ moves only $0.692\to0.709$
  ($+0.017$; per-cell median shift $+0.001$). The naive estimate and the Fano
  factor are unchanged ($p_{\mathrm{pair}}$ does not enter them). So the §4.5 / Figure-2
  / Figure-4 conclusions stand under a directly-estimated close-pair density.

- **Direction 2 and the gap are more sensitive — and `'direct'` is the more
  correct estimator there.** `central` (target $p^2$) moves $0.537\to0.477$
  ($-0.060$) and the full-vs-central gap widens $0.119\to0.183$. This is
  expected: $p_{\mathrm{pair}}$ enters `central` through the per-sample reweight, and
  `'direct'` makes `central` genuinely consistent over the *true* close-pair
  distribution rather than the idealized $p^2$.

On the strength of this comparison `'direct'` is now the estimator default
(2026-06-24), and the main writeup reports it throughout (§4.5). `'squared'`
remains available and reproduces the pre-change weights bit-for-bit; it is the
natural choice for the closed-form synthetic validation, where `p_pair = p^2`
holds exactly.

---

# 1. Where the $p^2$ assumption enters

The matched estimator (§4.2) reweights toward a target $q$ by importance
sampling. A single trial drawn from $p$ contributes weight $q(e)/p(e)$ to the
mean / $C_{\mathrm{total}}$ / $C_{\mathrm{psth}}$; a close **pair**, drawn from $p_{\mathrm{pair}}$, contributes $q(e)/p_{\mathrm{pair}}(e)$ to the rate second moment. The two
consistent directions instantiate this as

$$
\text{full } (q=p): \quad
w^{\mathrm{pair}} = \frac{p}{p_{\mathrm{pair}}}, \quad w^{\mathrm{samp}} = 1;
\qquad
\text{central } (q=p^2): \quad
w^{\mathrm{pair}} = \frac{p^2}{p_{\mathrm{pair}}}, \quad w^{\mathrm{samp}} = \frac{p_{\mathrm{pair}}}{p}.
\tag{1}
$$

The writeup never estimates $p_{\mathrm{pair}}$. It substitutes the §A.5 identity
$p_{\mathrm{pair}} = \hat p^2$, which collapses (1) to the implemented weights:

$$
\text{full}: \quad w^{\mathrm{pair}} = \frac{1}{\hat p}, \quad w^{\mathrm{samp}} = 1;
\qquad
\text{central}: \quad w^{\mathrm{pair}} = 1, \quad w^{\mathrm{samp}} = \hat p.
\tag{2}
$$

Note the asymmetry that answers "why would `central`'s weights change?": under
$p_{\mathrm{pair}}=\hat p^2$, `central`'s close-pair weight is exactly $1$ — and it
**stays** $1$ under a direct estimate, because `central`'s defining target *is*
the close-pair distribution, so its second moment never needs reweighting. What
the $p^2$ assumption controls for `central` is the **per-sample** weight $w^{\mathrm{samp}}$ that pulls $C_{\mathrm{total}}$, $C_{\mathrm{psth}}$ and the mean *in* to the
close-pair distribution. For `full` it is the close-pair weight $w^{\mathrm{pair}}$.
Each direction therefore probes one factor of $p_{\mathrm{pair}}$.

The §A.5 identity is exact for single-bin close pairs as $\Delta e\to0$. The
production estimator (§4.4) violates both premises: a finite threshold
$\varepsilon=0.05$, and a close-pair filter that operates on the whole 12-bin
RMS trajectory while the density is built on the reduced points $\rho_i$. There
is no theorem that $p_{\mathrm{pair}}=\hat p^2$ survives.

# 2. The `'direct'` estimator

`closepair_density='direct'` fits a second KDE $\hat p_{\mathrm{pair}}$ on the
realized close-pair representative-midpoints $\{\tfrac12(\rho_i+\rho_j)\}$ and
substitutes it for $\hat p^2$ in (1): `full`'s close-pair weight becomes $\hat
p/\hat p_{\mathrm{pair}}$, `central`'s per-sample weight becomes $\hat p_{\mathrm{pair}}/\hat p$. Because every estimator is self-normalized, the unknown $p^2$
normalizer cancels and only the *shapes* of $\hat p$ and $\hat p_{\mathrm{pair}}$
matter. When $\hat p_{\mathrm{pair}}=\hat p^2$ the weights reduce to (2) exactly, so
`'direct'` $\equiv$ `'squared'` wherever the identity holds.

**Validation (TDD).** Three tests in `test_estimators.py` pin this. In the
single-bin regime and the zero-drift trajectory limit — where $p_{\mathrm{pair}}=p^2$
is provable — `'direct'` recovers the flat-mask closed form $1-\alpha^p$ and
agrees with `'squared'` for Direction 1 (`test_direct_closepair_density_*`,
`test_trajectory_direct_closepair_density_*`); and an unqualified call is
confirmed to equal `'direct'` (`test_closepair_density_default_is_direct`).

# 3. How far is $\hat p_{\mathrm{pair}}$ from $\hat p^2$?

For an isotropic Gaussian $p=\mathcal N(0,\sigma_e^2 I)$ the §A.5 close-pair
density is exactly $\mathcal N(0,\tfrac{\sigma_e^2}{2}I)$ — variance halved — so
$\operatorname{tr}\operatorname{cov}(\rho_{\mathrm{mid}})/\operatorname{tr}
\operatorname{cov}(\rho)=0.5$. On the real sessions the population median is
**$0.42$** (range $0.23$–$0.85$ across sessions), with
$\mathrm{KL}(\hat p_{\mathrm{pair}}\,\|\,\hat p^2)$ median **$0.16$** (max $0.42$).
The departure is not noise: the real close-pair density is systematically
**more concentrated** than $p^2$ (Fig. A, B). Two mechanisms push it there — real
fixation densities are leptokurtic (a sharp central fixation peak plus a
microsaccade-driven tail), and the whole-window RMS-trajectory match keeps the
most stable, most central fixations — both of which over-represent the centre
relative to the Gaussian $p^2$ prediction. The wide across-session range tracks
cell yield: the high-yield Allen sessions sit at $0.28$–$0.38$ and dominate the
cell-weighted population, while the sparser Logan sessions scatter above $0.5$.

![**Close-pair density vs the squared marginal, and its effect on §4.5.**
**(A)** $x$-marginal of the eye density on a near-median session: $p$ over the
representative points (blue), the assumed $p^2=\mathcal N(0,\sigma^2/2)$ (red
dashed), and the directly-estimated close-pair density $\hat p_{\mathrm{pair}}$
(purple). $\hat p_{\mathrm{pair}}$ sits inside $p^2$ — real close pairs are more
central than the squared marginal predicts. **(B)** Per-session close-pair
variance ratio against the ideal Gaussian value $0.5$ (red), coloured by
$\mathrm{KL}(\hat p_{\mathrm{pair}}\,\|\,\hat p^2)$; population median $0.42$. **(C)**
Per-cell $1-\alpha$ for Direction 1 (`full`, target $p$): `'squared'` vs
`'direct'` close-pair density — on the identity line ($\Delta$median $+0.001$).
**(D)** Direction 2 (`central`, target $p^2$): systematically lower under
`'direct'` ($\Delta$median $-0.037$).](figures/fig_closepair_density.png)

# 4. Effect on the §4.5 results

Re-running `pipeline.decompose_session` under each close-pair-density estimate,
pooled over the same **1359** fig2-good cells (2 monkeys, 25 sessions):

| quantity | squared ($p^2$) | direct ($p_{\mathrm{pair}}$) | $\Delta$ median |
|---|---|---|---|
| median $1-\alpha$ naive | 0.728 | 0.728 | $+0.000$ |
| median $1-\alpha$ full ($p$) | 0.692 | 0.709 | $+0.017$ |
| median $1-\alpha$ central ($p^2$) | 0.537 | 0.477 | $-0.060$ |
| median Fano naive | 0.943 | 0.943 | $+0.000$ |
| median Fano full ($p$) | 0.954 | 0.953 | $-0.001$ |
| median \|full $-$ central\| gap | 0.119 | 0.183 | $+0.064$ |

- **`naive` and Fano are invariant** — a built-in control. The naive estimator
  uses no importance weights, so it cannot depend on the close-pair-density
  estimate, and indeed it does not move at all. The corrected Fano factor (a
  count-per-mean dominated by $C_{\mathrm{total}}$) shifts by $-0.001$.

- **Direction 1 (`full`, the reported number) is robust.** The population median
  rises by $+0.017$ and the per-cell median shift is $+0.001$ (per-cell $|\cdot|$
  median $0.035$, p90 $0.224$; Fig. C). The §4.5 headline — "matching to the
  actual viewing distribution leaves the reported $1-\alpha$ essentially
  unchanged" — survives the replacement of $\hat p^2$ by the directly-estimated
  $\hat p_{\mathrm{pair}}$.

- **Direction 2 (`central`) drops by $-0.060$** (per-cell median $-0.037$,
  $|\cdot|$ median $0.057$, p90 $0.186$; Fig. D). This is exactly where
  $p_{\mathrm{pair}}$ acts on `central` — the per-sample reweight — and the sign
  follows the variance ratio: a more-central close-pair density pulls the
  `central` decomposition further toward the fixation peak, where the rate is
  least eye-modulated, lowering $1-\alpha$.

- **The gap widens, $0.119\to0.183$.** `full` rises and `central` falls, so the
  full-vs-central gap — the fixation-scale spatial-structure measure of §4.3 —
  grows. Under `'direct'` it is a cleaner measure: both ends are now consistent
  over their true sampling distributions rather than over an idealized $p^2$.

# 5. Bottom line

The squared-marginal shortcut $p_{\mathrm{pair}}=\hat p^2$ is measurably imperfect in
the production trajectory-mode setting — real close pairs are more central than
$p^2$ by a population-median variance ratio of $0.42$ vs $0.5$. Estimating the
close-pair density directly removes the assumption. The reported Direction 1
$1-\alpha$ barely moves ($+0.017$ at the median, $+0.001$ per cell), so the
main-writeup conclusions are now robust to it *by measurement*, not by
appeal to the $\Delta e\to0$ identity. The more distribution-sensitive
quantities — Direction 2 and the full-vs-central gap — shift in the predicted
direction, and `'direct'` is the more principled estimator for them because it
targets the close-pair distribution the data actually sample. On this basis
`'direct'` is now the estimator default and the main writeup reports it
throughout; `'squared'` is retained for the closed-form synthetic validation,
where the $p_{\mathrm{pair}}=p^2$ identity holds exactly.
