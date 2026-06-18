---
title: "Variance decomposition with known rates: the one-way ANOVA"
subtitle: "A side note to `writeup.md` (methods_eyepos_matching)"
author: "fem-v1-fovea methods note"
date: "2026-05-31"
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

This is a side note to the main methodological writeup in this folder
(`writeup.md`, *Extending McFarland's cross-trial decomposition...*). It
covers a special case the main note does not address: what happens when
the rate $r(t, e)$ is **observed without Poisson noise** — the digital-twin
setting in `VisionCore/ryan/fig4/generate_fig4d.py`, where a deterministic
model emits $\hat r(t, e_i)$ at each trial's actual eye position.

When rates are deterministic, the Law-of-Total-Variance decomposition the
main note builds for spike counts collapses to a textbook one-way
random-effects ANOVA of $r$ grouped by analysis time bin $t$. Sections
1–4 record the math and identify what the ANOVA targets: exactly
$1-\alpha^p$, i.e. Direction 1 of `writeup.md` §4.2. Section 5 verifies
on the unified synthetic from `writeup.md` §2.3 that the ANOVA recovers
the analytical truth across all three mask kinds with visibly tighter bars
than the matched close-pair Direction-1 estimator.

Section 6 applies this to fig4 panel D and uncovers an unexpected residual:
the ANOVA and the close-pair Direction-1 estimator, which agree on the
synthetic, disagree by $0.08$ at the median when applied to the SAME
real twin $\hat r$. The downstream panel-D "twin is more eye-sensitive
than the neurons" appearance is fully an estimator-comparison artifact;
the twin reproduces the cells when the matched close-pair Direction-1
is used on both axes ($|\text{offset}| < 0.01$, $\rho = 0.55$–$0.63$).
**This is the reason the note is parked here rather than baked into the
main writeup — panel D in the paper currently uses the empirical pipeline
on both axes; the ANOVA-vs-D1 discrepancy on real $\hat r$ is a
side-finding worth recording but not load-bearing for the main story.**

References to the main writeup use the form `writeup.md §X.Y`.

---

# 1. The estimator

Group trials by time bin; bin $t$ has $n_t$ valid trials, $T$ kept bins,
$N = \sum_t n_t$. With per-bin mean $\bar r(t)$ and grand mean $\bar r$,

$$
\mathrm{SS}_\text{within}  = \sum_t \sum_i (r_{i,t} - \bar r(t))^2,
\qquad
\mathrm{SS}_\text{between} = \sum_t n_t\,(\bar r(t) - \bar r)^2,
$$

$$
\mathrm{MS}_\text{within} = \frac{\mathrm{SS}_\text{within}}{N - T},
\qquad
\mathrm{MS}_\text{between} = \frac{\mathrm{SS}_\text{between}}{T - 1}.
$$

The mean squares have exact expectations (method of moments — no Gaussian
assumption) $\mathbb E[\mathrm{MS}_\text{within}] = \sigma^2_W$ and
$\mathbb E[\mathrm{MS}_\text{between}] = \sigma^2_W + n_0\,\sigma^2_B$, with
unbalanced effective group size

$$
n_0 = \frac{N - \sum_t n_t^2 / N}{T - 1}
\qquad (= n \text{ when balanced}),
$$

yielding unbiased components

$$
\hat\sigma^2_W = \mathrm{MS}_\text{within},
\qquad
\hat\sigma^2_B = \max\!\Big(\frac{\mathrm{MS}_\text{between} - \mathrm{MS}_\text{within}}{n_0},\,0\Big),
$$

and FEM fraction
$\widehat{1-\alpha}_{\text{ANOVA}} = \hat\sigma^2_W / (\hat\sigma^2_W + \hat\sigma^2_B)$.
This is `VisionCore.covariance.rate_variance_components`, the model-side
function used in fig4 panel D.

# 2. What the ANOVA targets

Trial-level eye positions at every time bin are drawn from the marginal
viewing distribution $p$, so both the within-bin variance and the
across-(t, trial) total variance live under $e \sim p$. Working through the
unified rate field (`writeup.md` Eq. 9; let
$G_t^p = \mathbb E_{e\sim p}[M(e)\,s_t(e)]$, so
$\mathbb E_s[(G_t^p)^2] = I_{M,K,p}$ as in `writeup.md` §A.5):

$$
\mathbb E_t\!\big[\mathrm{Var}_{e\sim p}(r\,\vert\, t)\big]
 = \mathbb E_w[\alpha^2]\,\big(\tau^2\,\mathbb E_p[M^2] - I_{M,K,p}\big),
$$

$$
\mathrm{Var}_t\!\big(\mathbb E_{e\sim p}[r\,\vert\, t]\big)
 = \mathbb E_w[\alpha^2]\,I_{M,K,p},
$$

$$
\mathrm{Var}_{t,\,e\sim p}(r)
 = \mathbb E_w[\alpha^2]\,\tau^2\,\mathbb E_p[M^2],
$$

where $\mathbb E_t$ denotes the joint expectation over the per-bin field
draw $s_t$ and across-bin sampling, and the $n_t$-weighting in
$\mathrm{SS}_\text{between}$ corresponds to the pair-count time-bin
weighting of `writeup.md` §3. Hence in the large-$(N, T)$ limit

$$
\boxed{\;
\widehat{1-\alpha}_{\text{ANOVA}}
 \;\longrightarrow\;
 1 - \frac{I_{M,K,p}}{\tau^2\,\mathbb E_p[M^2]}
 \;=\; 1-\alpha^p,
\;}
$$

which is **exactly Direction 1** of `writeup.md` §4.4. The reason is
sampling: the ANOVA's reference distribution is the empirical viewing
distribution, and that is $p$, not $p^2$. The close-pair filter is what
would have introduced $p^2$, and the ANOVA does not use one.

# 3. The $1/n_t$ inflation, with or without (A1)

Even with exact rates, the naive between-bin variance
$\mathrm{Var}_t(\bar r(t))$ is biased up by
$\mathbb E_t[\mathrm{Var}_e(r\,\vert\,t)/n_t]$: each time-bin mean averages
only $n_t$ eye draws of the same field. The
$(\mathrm{MS}_\text{between} - \mathrm{MS}_\text{within})/n_0$ subtraction
removes that term exactly. The bias is structural — it appears under
balanced $n_t$ too (then $n_0 = n$ and the inflation is the familiar $1/n$
shrinkage) — and worst precisely for FEM-dominated cells, where
$\mathrm{Var}_e(r\,\vert\,t)$ is large. It is the ANOVA's analogue of the
distinct-trial trick in `writeup.md` §A.7: both pull the within-bin
sampling noise out of the between-bin moment without needing two trials
at the same eye position.

# 4. Time-bin weighting under variable $n_t$

The ANOVA's $n_t$ entries — $n_t$ in $\mathrm{SS}_\text{between}$,
$(n_t{-}1)$ degrees of freedom per bin in $\mathrm{MS}_\text{within}$, and
the $n_t$-weighted grand mean $\bar r$ — give it an effective time-bin
weighting $w_t \propto n_t$. This is a third point on `writeup.md` §3's
weighting axis, distinct from the uniform ($w_t \propto 1/T$) and pair-count
($w_t \propto n_t(n_t{-}1)/2$) directions used by the literal McFarland
$C_\text{psth}$ and the close-pair $C_\text{rate}$ respectively. The
invariance result of `writeup.md` §A.5 applies regardless: $w_t$
multiplies both $\hat\sigma^2_W$ and $\hat\sigma^2_B$ through
$\mathbb E_w[\alpha^2]$ and cancels in the ratio.

To match this weighting on the close-pair side, `estimators.decompose`
exposes `time_bin_weighting='trial_count'` (per-sample weight 1; per-pair
weight $n_t/m_t$ where $m_t$ is the close-pair count in bin $t$).

The residual (A1) bias on the ANOVA comes from the $(n_t{-}1)$ vs $n_t$
mismatch between $\hat\sigma^2_W$ and $\hat\sigma^2_B$: when $\alpha(t)$ is
correlated with $n_t$, the two terms pick out slightly different envelope
averages. The mismatch is $O(1/\min_t n_t)$ on the ratio — tiny at the
$n_t \sim 10^2$ scale of fixRSVP fixations per analysis time bin, and zero
for any cell with $\alpha(t)$ uncorrelated with $n_t$. By comparison the
close-pair pipeline's (A1) bias is first-order in the $\alpha$–$n_t$
covariance, because its numerator and denominator use weightings that
scale quadratically vs linearly in $n_t$ (the motivating bias of
`writeup.md` §3.3 and `fig_time_bin_weighting.py`).

Operationally, "variable trial length" in the fig4d / fixRSVP setting
means variable fixation duration: a 400 ms fixation contributes valid data
at all $\sim$48 within-fixation time bins (at 120 Hz), a 200 ms one only at
the first $\sim$24. The ANOVA sees this as the per-bin valid count $n_t =$
number of fixations long enough to reach bin $t$, decaying roughly
geometrically with $t$. `rate_variance_components` reads $r$ and the
validity mask and proceeds — no explicit reweighting required.

# 5. Validation across masks

Figure 1 sweeps $\ell/\sigma$ over $[0.25, 4]$ on the unified synthetic
(`writeup.md` §2.3) at each of the three mask kinds and overlays, per
panel:

1. **Analytical $1-\alpha^p$** — closed form via `writeup.md` §A.1–A.4
   (flat, central, eccentric all close analytically) via
   `synthetic.ground_truth`;
2. **ANOVA on noise-free rates** — `rate_variance_components` on the
   deterministic rate field of one cell;
3. **Direction-1 close-pair on the same data** —
   `decompose(target='full', density='gaussian')` from this folder
   (importance-reweighted close-pair second moment).

All three agree across all three masks. ANOVA markers sit tightly on the
analytical curve; the close-pair Direction-1 markers are unbiased but
visibly wider, especially for the **eccentric** mask, where the FEM
variance lives in the periphery and the $1/\hat p(e)$ weights pick up
unbounded tails. The ANOVA pays no such tax: it uses every
$(\text{trial}, \text{time bin})$ sample once, with no $\Delta e$ threshold
and no importance weights.

![**Figure 1.** ANOVA on noise-free rates recovers the analytical $1-\alpha^p$ across all three mask kinds, agreeing with the matched close-pair Direction-1 estimator (`decompose(target='full')`) but with visibly tighter error bars. Same target, two estimators; ANOVA wins on efficiency when rates are deterministic.](figures/fig_anova.png)

# 6. Application to fig4 panel D

Fig4 panel D plots model 1-α (from `rate_variance_components` on the twin's
deterministic rates $\hat r(t, e)$) against empirical 1-α (from the
production close-pair pipeline on real spikes). The published panel shows
the model running over-eye-sensitive: a median twin-vs-cell offset of
$+0.083$ (twin higher; see Figure 2 A). §2 places ANOVA at Direction 1,
so swapping the empirical side to a matched Direction-1 close-pair should
close the gap. It does not — the cell-side D1 close-pair sits $0.04$–$0.09$
*below* the twin ANOVA depending on time-bin weighting, with the gap
widening as cell-side weighting matches the ANOVA's $w_t \propto n_t$
(Figure 2 A is the `trial_count`-matched case).

The diagnostic is Figure 2 B: the same close-pair Direction-1 estimator
applied to the twin's $\hat r(t, e)$ disagrees with the ANOVA on the same
rates by $0.08$ at the median, with Spearman $\rho = 0.49$ — two
estimators that §2 says target the same population $1-\alpha^p$ and §5
confirmed agree on synthetic, diverging systematically on real fixRSVP
$\hat r$. The 0.08 ANOVA-vs-D1-on-rhat offset is exactly the gap of panel
A. The cell-vs-twin question is downstream of an estimator-comparison
question that the synthetic validation of §5 did not expose.

Holding the estimator fixed across the two axes removes the dependence on
which of ANOVA / close-pair D1 is "right" on real $\hat r$. Figure 2 C
and D pair the empirical close-pair D1 (on real spikes) against the same
close-pair D1 on the twin $\hat r$, both with the same time-bin
weighting. Under matched estimators the twin reproduces the cells:
pair_count median twin-vs-cell offset $-0.005$ with $\rho = 0.63$,
trial_count $+0.010$ with $\rho = 0.55$. The "twin is more eye-sensitive
than the neurons" appearance of panel A is an estimator-method artifact,
not a real model property.

For the paper, panel D uses the empirical pipeline on both axes
(`pipeline_one_minus_alpha` from `VisionCore/covariance.py`, applied to
real spikes for the x-axis and to $\hat r$ for the y-axis). This avoids
the ANOVA-vs-D1 disagreement on real $\hat r$ entirely. The ANOVA result
of §2 is still mathematically correct and useful as a model-side
diagnostic on synthetic, but the on-real-$\hat r$ ANOVA-vs-close-pair
divergence is a methodological loose end: both estimators target the same
population, both recover the truth on synthetic, yet they sit $0.08$
apart at the per-cell level on real $\hat r$. Plausible suspects —
sharper variable $n_t$ than the synthetic sweep covered, KDE tails in
real eye distributions, the $\Delta e < 0.05^\circ$ threshold selecting a
different effective sub-population at finite $N$ — are not isolated here;
the question is flagged for a future sweep.

![**Figure 2 — Panel D under matched estimators.** **(A)** Reference:
empirical close-pair Direction-1 (`time_bin_weighting='trial_count'`,
matching ANOVA's effective $w_t \propto n_t$) vs twin ANOVA — the
$0.08$-at-median twin-over-cells gap. **(B)** Same twin $\hat r$, two
estimators: close-pair Direction-1 (trial_count) vs ANOVA. The estimators
that agree on synthetic (Figure 1) disagree by $0.08$ at the median on
real $\hat r$, accounting for the entire panel-A gap. **(C, D)** Matched
estimator on both axes: close-pair Direction-1 (`pair_count` and
`trial_count` respectively) on cells (x) and twin (y). Under either
matched weighting the twin reproduces the cells at the median (offset
$|{<}0.01|$, $\rho = 0.55$–$0.63$). The panel-A apparent "twin
over-sensitivity" is an estimator-comparison artifact; the underlying
twin-cell agreement on $1-\alpha^p$ is good.](figures/fig_panel_d_closepair.png)

---

*Reproduce figures and the synthetic ANOVA validation (from this folder):*

```bash
uv run python fig_anova.py                 # Figure 1 (known-rate ANOVA validation)
uv run python fig_panel_d_anova.py         # cell-side matching, ANOVA on twin
uv run python fig_panel_d_closepair.py     # Figure 2 (matched close-pair on both sides)
```

*Build this note to a self-contained, offline HTML (math via MathML, images
inlined):*

```bash
pandoc note_anova.md -s --mathml --self-contained -o note_anova.html
```
