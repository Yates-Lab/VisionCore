---
title: "Single-trial variance explained relative to explainable rate variance"
subtitle: "Definition, derivation, relation to CCnorm, and interpretation for Figure 3"
author: "fem-v1-fovea methods note"
date: "2026-07-28"
header-includes: |
  <style>
  .numbered-equation { position: relative; }
  .numbered-equation .eqno {
    position: absolute; right: 0.5em; top: 50%;
    transform: translateY(-50%);
  }
  table { border-collapse: collapse; margin: 1.2em 0; font-size: 0.88em; }
  th, td { border-bottom: 1px solid #ddd; padding: 0.3em 0.7em; text-align: left; }
  th { border-bottom: 2px solid #999; }
  code { font-size: 0.88em; }
  blockquote { border-left: 3px solid #ccc; padding-left: 1em; color: #444; }
  </style>
---

# Summary

Single-trial spike counts contain two kinds of variation. The conditional firing
rate changes with the stimulus and gaze, while spiking and other response
fluctuations produce variation around that rate. A deterministic model can
predict the first component but cannot predict the second from stimulus and
gaze alone. Consequently, a low single-trial $r^2$ can reflect either an
inaccurate model or a response dominated by unpredictable variation.

The normalized score separates these possibilities:

$$
Q \;=\; \frac{r^2}{R^2_{\max}},
\qquad
r^2 = 1-\frac{\operatorname{Var}(Y-\hat Y)}{\operatorname{Var}(Y)},
\qquad
R^2_{\max}=\frac{\operatorname{Var}(\lambda)}{\operatorname{Var}(Y)},
\tag{1}
$$

where $Y$ is the spike count in one 120 Hz bin, $\hat Y$ is the model prediction,
and $\lambda=\mathbb E[Y\mid Z]$ is the conditional mean count given the
variables $Z$ available to the predictor. When all quantities refer to the same
conditioning variables and sampling distribution,

$$
Q
=1-\frac{\operatorname{Var}(\lambda-\hat Y)}
         {\operatorname{Var}(\lambda)}.
\tag{2}
$$

Thus, $Q$ is the fraction of conditional rate variance explained by the model,
using the same variance-based definition as the implemented single-trial
$r^2$. It has a natural constant-rate reference of zero and an oracle reference
of one. Negative values indicate that the model's rate error varies more than
the true conditional rate. Values above one can occur for the empirical score
because the denominator is estimated with finite data and conditions on a
shorter gaze history than the model. Section 4 gives the estimator the figure
computes, which divides the captured count variance by the rate variance Figure
2 measures.

This metric complements $\mathrm{CC}_{\mathrm{norm}}$. The latter asks whether
the model predicts the shape of the trial-averaged response across stimulus
time after correcting for the reliability of the measured trial average. It
discards trial-specific variation. The normalized single-trial $r^2$ asks how
much stimulus- and gaze-dependent rate variation the model predicts on
individual trials. Together, the two metrics distinguish accurate prediction of
the average response from accurate prediction of the structured variation
around that average.

# 1. Statistical target

Consider one neuron. Let $I$ index trials and $T$ index 8.33 ms time bins within
the repeated fixated-flashed-image sequence. Sampling a valid pair $(I,T)$
defines a random spike count

$$
Y=Y_{I,T}.
\tag{3}
$$

Let $Z$ contain the information used to predict that count. For the conceptual
derivation, $Z$ includes stimulus time and the relevant gaze trajectory. In the
experiment, stimulus time identifies the screen image because the same image
sequence is repeated across trials. Gaze determines how that screen image moves
across the retina. The digital twin predicts a mean count from a gaze-contingent
retinal stimulus history and eye-position-derived behavioral covariates, so a
fixed model produces

$$
\hat Y=g(Z).
\tag{4}
$$

The best possible squared-error prediction based on $Z$ is the conditional
mean

$$
\lambda(Z)=\mathbb E[Y\mid Z].
\tag{5}
$$

The conditional mean is a property of the neural response distribution rather
than of the fitted twin. It represents the mean count that would be observed if
the same stimulus and gaze state could be repeated many times. Write the
observed count as

$$
Y=\lambda(Z)+\varepsilon,
\qquad
\mathbb E[\varepsilon\mid Z]=0.
\tag{6}
$$

The residual $\varepsilon$ includes stochastic spiking and any response
variation not predictable from $Z$. Equation (6) does not assume a Poisson
distribution. The conditional residual has mean zero by the definition of
$\lambda$.

All variances below are taken over the joint distribution of valid trials and
time bins for one neuron. The numerator and denominator must use the same bin
width and, for the exact interpretation, the same sampling distribution.

# 2. Explainable and residual variance

The law of total variance divides the observed count variance into rate
variance and conditional residual variance:

$$
\operatorname{Var}(Y)
=
\operatorname{Var}\!\left(\mathbb E[Y\mid Z]\right)
+
\mathbb E\!\left[\operatorname{Var}(Y\mid Z)\right].
\tag{7}
$$

Substituting $\lambda=\mathbb E[Y\mid Z]$ gives

$$
\operatorname{Var}(Y)
=
\underbrace{\operatorname{Var}(\lambda)}_{\text{rate variance}}
+
\underbrace{\mathbb E[\operatorname{Var}(Y\mid Z)]}_{\text{conditional residual variance}}.
\tag{8}
$$

This identity can also be obtained directly from (6). Because
$\mathbb E[\varepsilon\mid Z]=0$, the residual is uncorrelated with every
function of $Z$, including $\lambda$:

$$
\begin{aligned}
\operatorname{Cov}(\varepsilon,\lambda)
&=\mathbb E[\varepsilon\lambda]
   -\mathbb E[\varepsilon]\mathbb E[\lambda] \\
&=\mathbb E\!\left[
   \lambda\,\mathbb E[\varepsilon\mid Z]
   \right] \\
&=0.
\end{aligned}
\tag{9}
$$

Therefore $\operatorname{Var}(Y)=\operatorname{Var}(\lambda)+
\operatorname{Var}(\varepsilon)$. The residual variance equals the expected
conditional variance because its conditional mean is zero:

$$
\operatorname{Var}(\varepsilon)
=\mathbb E[\varepsilon^2]
=\mathbb E[\operatorname{Var}(Y\mid Z)].
\tag{10}
$$

Only $\operatorname{Var}(\lambda)$ is predictable from $Z$. This motivates the
single-trial variance ceiling

$$
R^2_{\max}
=
\frac{\operatorname{Var}(\lambda)}{\operatorname{Var}(Y)}.
\tag{11}
$$

For example, $R^2_{\max}=0.12$ means that 12% of the variance in 8.33 ms spike
counts reflects systematic rate modulation under the chosen conditioning
variables. The remaining 88% is conditional residual variance at that temporal
resolution. A perfect rate model can reach an observed-count $r^2$ of only
0.12, even though it predicts all rate modulation.

# 3. Derivation for the implemented single-trial $r^2$

The figure computes

$$
r^2
=1-\frac{\operatorname{Var}(Y-\hat Y)}{\operatorname{Var}(Y)}.
\tag{12}
$$

This variance form differs slightly from the common sum-of-squared-errors form.
It is the relevant definition because the plotted numerator is the difference
between the two variances it contains, taken over one common set of bins per
unit by `compute_matched_captured_variance` in
`paper/fig3/_fig3_explainable_variance.py`.

Using $Y=\lambda+\varepsilon$, the prediction error is

$$
Y-\hat Y
=\varepsilon+(\lambda-\hat Y).
\tag{13}
$$

Both $\lambda$ and $\hat Y$ are functions of $Z$. The conditional residual is
therefore uncorrelated with their difference:

$$
\begin{aligned}
\operatorname{Cov}\!\left(\varepsilon,\lambda-\hat Y\right)
&=\mathbb E\!\left[
(\lambda-\hat Y)\,\mathbb E[\varepsilon\mid Z]
\right] \\
&=0.
\end{aligned}
\tag{14}
$$

The variance of the prediction error consequently separates into two terms:

$$
\operatorname{Var}(Y-\hat Y)
=
\mathbb E[\operatorname{Var}(Y\mid Z)]
+
\operatorname{Var}(\lambda-\hat Y).
\tag{15}
$$

The first term is irreducible for a predictor based on $Z$. The second is the
model's error in predicting the conditional rate. Substitution into (12),
followed by use of (8), yields

$$
\begin{aligned}
r^2
&=1-
\frac{
\mathbb E[\operatorname{Var}(Y\mid Z)]
+\operatorname{Var}(\lambda-\hat Y)
}{\operatorname{Var}(Y)} \\
&=\frac{
\operatorname{Var}(\lambda)
-\operatorname{Var}(\lambda-\hat Y)
}{\operatorname{Var}(Y)} \\
&=R^2_{\max}
-\frac{\operatorname{Var}(\lambda-\hat Y)}{\operatorname{Var}(Y)}.
\end{aligned}
\tag{16}
$$

Dividing by (11) gives the normalized score:

$$
\boxed{
\frac{r^2}{R^2_{\max}}
=1-
\frac{\operatorname{Var}(\lambda-\hat Y)}
     {\operatorname{Var}(\lambda)}
}.
\tag{17}
$$

Equation (17) is an ordinary variance-explained coefficient with the latent
conditional rate as its target. Normalization removes the conditional response
noise from the scale without removing it from the data used to score the model.

The population quantity has the following interpretation:

| value | interpretation |
|---|---|
| $1$ | The prediction matches all variation in $\lambda$, up to a constant offset. |
| between $0$ and $1$ | The prediction reduces rate-error variance relative to a constant prediction. |
| $0$ | The prediction explains no conditional rate variance. Any constant prediction has this value. |
| below $0$ | Rate-error variance exceeds $\operatorname{Var}(\lambda)$. |
| above $1$ | Impossible for exact population quantities defined on the same $Z$ and distribution; possible for the empirical ratio because its denominator is estimated. |

Because (12) uses residual variance, a constant error is not penalized. The
maximum is attained whenever $\hat Y=\lambda+c$. An MSE-based coefficient would
instead satisfy

$$
r^2_{\mathrm{MSE}}
=r^2-
\frac{\left(\mathbb E[Y-\hat Y]\right)^2}{\operatorname{Var}(Y)},
\tag{18}
$$

and would attain its maximum only at $c=0$. Predictions in Figure 3 undergo a
per-unit affine calibration before scoring, which makes mean errors small, but
the plotted metric remains the variance-based quantity in (12).

# 4. Estimating the denominator from the covariance decomposition

The conditional rate $\lambda$ is not observed directly. Figure 2 estimates its
variance from repeated presentations and matched gaze trajectories. For each
unit, the diagonal entries of the covariance decomposition provide

$$
\operatorname{diag}C_{\mathrm{total}}
\approx \operatorname{Var}(Y),
\qquad
\operatorname{diag}C_{\mathrm{rate}}
\approx \operatorname{Var}(\lambda).
\tag{19}
$$

Figure 3 uses the second of these directly. For each unit it divides the
captured count variance by Figure 2's rate variance,

$$
\widehat Q
=
\frac{\operatorname{Var}(Y)-\operatorname{Var}(Y-\hat Y)}
     {\operatorname{diag}C_{\mathrm{rate}}}.
\tag{20}
$$

By (16) the numerator estimates $\operatorname{Var}(\lambda)-
\operatorname{Var}(\lambda-\hat Y)$, so (20) targets the same population
quantity as (17). Dividing the measured $r^2$ by the empirical normalizer
$\widehat R^2_{\max}=\operatorname{diag}C_{\mathrm{rate}}/
\operatorname{diag}C_{\mathrm{total}}$ returns the same value whenever the
numerator's $\operatorname{Var}(Y)$ equals $\operatorname{diag}
C_{\mathrm{total}}$. The two forms differ by the ratio of those totals, which
Section 7 quantifies. Equation (20) takes one quantity from Figure 2 and leaves
the rest of the calculation on the sample the model was scored on.

## 4.1 Total count variance

$C_{\mathrm{total}}$ is the covariance of the observed spike counts pooled over
valid trials and stimulus times. Its diagonal is the variance of each unit's
single-bin counts. It contains stimulus-locked rate modulation,
gaze-dependent rate modulation, and conditional residual variance.

## 4.2 Conditional rate variance from close trial pairs

Suppose two distinct trials $i\ne j$ reach the same stimulus time with matching
gaze trajectories. Their counts can be written

$$
Y_i=\lambda_i+\varepsilon_i,
\qquad
Y_j=\lambda_j+\varepsilon_j.
\tag{21}
$$

If their gaze-conditioned rates are equal and their conditional residuals are
independent across trials, then

$$
\mathbb E[Y_iY_j\mid Z_i\approx Z_j]
\approx \lambda^2.
\tag{22}
$$

Cross-trial multiplication removes private response noise in expectation. After
averaging these products and subtracting the squared mean rate, the close-pair
estimator recovers $\operatorname{Var}(\lambda)$. Figure 2 uses gaze-trajectory
pairs within $0.05^\circ$ RMS distance.

Close pairs occur most often near the center of the fixation distribution.
Without correction, they sample approximately from the squared gaze density
rather than from the distribution of positions actually viewed. The production
estimator importance-weights close-pair midpoints back to the full viewing
distribution before computing $C_{\mathrm{rate}}$. This correction is needed so
that the numerator and denominator of (20) refer to the same gaze distribution.

## 4.3 Temporal resolution

$R^2_{\max}$ depends on counting-window duration. Longer windows average
conditional response noise and therefore have larger explainable fractions. The
twin predicts one count at 120 Hz, so Figure 3 uses the one-bin estimate from
Figure 2:

$$
\text{window}=1/120\ \mathrm{s}=8.33\ \mathrm{ms}.
\tag{23}
$$

A ceiling estimated from the 25 ms window displayed in Figure 2 would not be a
valid normalizer for the model's 8.33 ms single-trial $r^2$.

Units with non-positive estimated rate variance have an undefined denominator
and are excluded rather than clipped. Small positive estimates can produce
unstable ratios, so population summaries should use medians and display the
distribution rather than rely on its mean.

## 4.4 Which sample the denominator is estimated on

Figure 3 scores only the bins at which the twin produces a prediction. A
33-frame retinal history and the per-unit data filters remove the beginning of
every trial and scattered later bins, so the scored sample is a subset of the
one Figure 2 decomposes. Estimating $\operatorname{diag}C_{\mathrm{rate}}$ on
that subset would place the numerator and the denominator on a single sampling
distribution, which is the condition under which (17) holds exactly.

That estimate is not usable in practice. The close-pair estimator needs pairs of
distinct trials that reach the same stimulus time with matching gaze, and the
history mask removes most of them. Across the 994-unit Figure 2 inclusion
population it returns a positive rate variance for 509 units, against 972 for
the estimate Figure 2 reports. Where both exist they disagree by a factor
ranging from 0.10 to 5.19 between the 5th and 95th percentiles, with a median
of 1.11: the subset estimate is not biased in a consistent direction, it is
imprecise. In five of the nineteen sessions it returns nothing at all. Those
five are not sessions the twin predicts poorly; they are sessions where the
estimator's requirement of ten trials per time bin fails once the sample is
reduced.

The denominator is therefore Figure 2's own estimate, computed over every valid
bin of its $0.5^\circ$ fixation frame. Each unit keeps one denominator across
all model conditions, and that denominator is the same number figure 2 and the
FEM-modulation panel report. The cost is that the numerator and the denominator
no longer describe the same sample, which Section 7 quantifies.

# 5. Relationship to $\mathrm{CC}_{\mathrm{norm}}$

Both metrics account for neural response variability, but they evaluate
different targets.

## 5.1 What $\mathrm{CC}_{\mathrm{norm}}$ measures

Let

$$
\mu(t)=\mathbb E[Y\mid T=t]
\tag{24}
$$

be the trial-averaged neural response, or ideal PSTH. For each eligible stimulus
time, the implementation averages observed counts and model predictions over
trials and computes

$$
\mathrm{CC}_{\mathrm{abs}}
=\operatorname{Corr}_t\!\left(\bar Y(t),\overline{\hat Y}(t)\right).
\tag{25}
$$

The measured PSTH is noisy because it is estimated from finitely many trials.
The implementation repeatedly splits trials in half, correlates the two
half-PSTHs, and averages those correlations to obtain $\mathrm{CC}_{\mathrm{half}}$.
The Spearman--Brown conversion estimates the reliability of the full PSTH:

$$
\mathrm{CC}_{\max}
=
\sqrt{\frac{2\,\mathrm{CC}_{\mathrm{half}}}
           {1+\mathrm{CC}_{\mathrm{half}}}}.
\tag{26}
$$

The reported score is

$$
\mathrm{CC}_{\mathrm{norm}}
=
\frac{\mathrm{CC}_{\mathrm{abs}}}{\mathrm{CC}_{\max}}.
\tag{27}
$$

This correction estimates how well the model's average response follows the
latent, noise-free PSTH. Pearson correlation measures temporal shape, not
calibrated amplitude. It is unchanged by a positive affine transformation of
the prediction. Averaging over trials also removes the trial-specific gaze
variation that motivates the single-trial analysis.

## 5.2 The PSTH occupies only part of the conditional rate variance

The conditional rate depends on stimulus time and gaze. Applying the law of
total variance to $\lambda(T,E)$ while conditioning on time gives

$$
\operatorname{Var}(\lambda)
=
\underbrace{\operatorname{Var}_T\!\left(\mu(T)\right)}_{\text{stimulus-locked rate variance}}
+
\underbrace{\mathbb E_T\!\left[
\operatorname{Var}_E(\lambda\mid T)
\right]}_{\text{FEM-dependent rate variance}}.
\tag{28}
$$

These are the diagonal terms called $C_{\mathrm{PSTH}}$ and
$C_{\mathrm{FEM}}$ in Figure 2, with
$C_{\mathrm{rate}}=C_{\mathrm{PSTH}}+C_{\mathrm{FEM}}$. Define the
stimulus-locked share of rate variance as

$$
\alpha
=
\frac{\operatorname{Var}(\mu)}{\operatorname{Var}(\lambda)}
=
\frac{\operatorname{diag}C_{\mathrm{PSTH}}}
     {\operatorname{diag}C_{\mathrm{rate}}}.
\tag{29}
$$

An ideal PSTH predictor sets $\hat Y=\mu(T)$. Its rate-prediction error is
$\lambda-\mu$, whose variance is the FEM-dependent term in (28). Equation (17)
then gives

$$
\left.\frac{r^2}{R^2_{\max}}\right|_{\hat Y=\mu}
=
1-
\frac{\operatorname{Var}(\lambda-\mu)}
     {\operatorname{Var}(\lambda)}
=
\alpha.
\tag{30}
$$

Thus, $\alpha$ is the maximum normalized single-trial score available to an
ideal predictor restricted to stimulus time. A model can exceed this reference
only by predicting rate modulation beyond the trial average. The measured
leave-one-out PSTH generally scores below $\alpha$ because it estimates
$\mu(t)$ from a finite number of trials.

## 5.3 The metrics answer complementary questions

| metric | target | what is averaged away? | normalization | main interpretation |
|---|---|---|---|---|
| $\mathrm{CC}_{\mathrm{norm}}$ | Trial-averaged response $\mu(t)$ | Trial-specific variation | Reliability of the measured PSTH | Accuracy of the average temporal response shape |
| Single-trial $r^2$ | Observed counts $Y_{i,t}$ | Nothing | Total observed count variance | Predictive variance on the raw count scale |
| $r^2/R^2_{\max}$ | Conditional rate $\lambda(t,e)$ | Conditional residual noise is removed from the scale | Estimated explainable rate fraction | Fraction of stimulus- and gaze-dependent rate variance predicted |

$\mathrm{CC}_{\mathrm{norm}}=0.7$ does not mean that 70% of explainable
single-trial variance is captured. Correlation is insensitive to response scale,
and it targets $\mu(t)$ rather than $\lambda(t,e)$. Squaring
$\mathrm{CC}_{\mathrm{norm}}$ does not resolve these differences. A model can
score well on $\mathrm{CC}_{\mathrm{norm}}$ by predicting the average response
while missing gaze-dependent trial variation. Conversely, a trial-specific
model can improve normalized single-trial $r^2$ with little change in
$\mathrm{CC}_{\mathrm{norm}}$ if the added predictions average to zero across
trials.

# 6. Contribution to the model analysis

Figure 2 establishes that much of the measurable rate variance in foveal V1 is
associated with fixational eye movements. Figure 3 asks whether an
image-computable model predicts that variation and which model input carries
it.

Raw single-trial $r^2$ establishes prediction against observed spikes, but its
absolute magnitude depends strongly on each unit's conditional residual noise.
On the scored bins the full twin has a median raw $r^2$ of $0.034$, against a
median $\widehat R^2_{\max}$ of $0.124$. The median of (20) is $0.269$: the full
twin recovers a little over a quarter of the rate variance Figure 2 identifies
as measurable at the model's temporal resolution. The leave-one-out PSTH reaches
$0.157$ on the same units, and freezing the retinal image drops the twin to
$0.042$. These numbers refer to the 972 units of the Figure 2 inclusion
population that have a positive one-bin rate variance, in nineteen sessions;
values elsewhere in the manuscript may differ when a different reliability
population or PSTH treatment is used.

The denominator is estimated from the neural data and is fixed across all model
conditions for a given unit. It therefore cannot make a weakened model appear
better by lowering its own reference. Within a unit, normalization is a
positive rescaling and cannot reverse the ordering of the full, ablated, and
stabilized predictions. Its contribution is to place neurons with different
noise levels on a common rate-variance scale before population aggregation.

The model perturbations then support three distinct conclusions:

1. **High $\mathrm{CC}_{\mathrm{norm}}$** shows that the twin predicts the
   held-out trial-averaged response.
2. **Single-trial $r^2/R^2_{\max}$ above the measured PSTH score** shows that the
   twin predicts structured rate variation not available in a finite trial
   average.
3. **Preservation under behavioral-input ablation and loss under retinal
   stabilization** localizes that trial-specific prediction within the twin to
   the moving retinal image rather than to the separate extraretinal pathway.

The third result is a within-model intervention. It shows which input route the
fitted twin uses and connects its behavior to the empirical covariance
decomposition. It does not by itself prove that the biological circuit lacks
extraretinal modulation.

# 7. Limits of the empirical interpretation

Equation (17) is exact for fixed predictors and population quantities defined
with the same $Z$ and sampling distribution. The plotted ratio replaces
$\operatorname{Var}(\lambda)$ with an estimate from a separate analysis
pipeline. Several limits follow.

**Conditioning-history mismatch.** The close-pair estimator matches a short
gaze trajectory, whereas the twin receives a 33-frame retinal history and a
broader eye-velocity embedding. Conditioning on more predictive information can
only increase $\operatorname{Var}(\mathbb E[Y\mid Z])$. The Figure 2 estimate
can therefore be lower than the ceiling appropriate to the model. Finite gaze
matching tolerance has the same likely effect. These differences allow an
empirical ratio above one without implying prediction beyond a true oracle.

**Estimator uncertainty.** $C_{\mathrm{rate}}$ is estimated from a limited
number of close trial pairs. Sampling error can make its diagonal small or
negative. Non-positive estimates are treated as undefined, while small positive
values create a long right tail in the ratio. Medians and quantiles are more
stable summaries than means. Using every bin Figure 2 accepts, rather than the
scored subset, is what keeps this term tolerable; Section 4.4 gives the size of
the difference.

**Cross-trial dependence.** The close-pair product removes conditional residual
noise only when residuals are independent across the paired trials. Slow changes
in excitability, adaptation, or recording stability can correlate residuals
across trials and inflate the estimated rate variance. This would increase
$\widehat R^2_{\max}$ and reduce the normalized model score.

**Sampling differences.** The numerator is measured on the counting windows of
Figure 2's $0.5^\circ$ fixation frame, restricted to the bins the twin can
predict; the denominator uses every bin that frame accepts. The scored sample
therefore carries less count variance than Figure 2 records: the ratio of the
two totals has a median of $0.899$ and an interquartile range of $0.776$ to
$0.992$. Equation (20) understates the captured fraction to that extent, and
normalizing by $\widehat R^2_{\max}$ instead would raise each unit's score by
the reciprocal of its own ratio. The figure reports the median ratio alongside
the panel rather than correcting for it, because the correction is itself a
per-unit estimate.

**In-sample calibration.** Each condition receives a per-unit affine rescaling
fit by Poisson likelihood on the responses that are subsequently scored. This
places model conditions on a comparable count scale but introduces slight
in-sample optimism. Cross-fitting the calibration would remove that optimism.

**Conditional rather than universal explainability.** The denominator measures
rate variance explained by the variables and trajectory resolution represented
in Figure 2. Residual variance can include unmeasured neural state as well as
stochastic spiking. The metric should therefore be described as single-trial
$r^2$ relative to the empirically measured stimulus- and gaze-conditional rate
variance, rather than as a universal fraction of all biologically predictable
activity.

# 8. Reporting definition

A concise methods definition is:

> Single-trial prediction was quantified per unit as the captured count
> variance, $\operatorname{Var}(Y)-\operatorname{Var}(Y-\hat Y)$, over the
> 8.33 ms counting windows of the Figure 2 fixation frame at which the model
> produced a valid prediction, divided by that unit's
> $\operatorname{diag}C_{\mathrm{rate}}$ from the Figure 2 covariance
> decomposition at the same counting window. Both variances were taken over one
> common set of bins per unit, shared by every scored condition. Under matched
> conditioning and sampling the ratio equals
> $1-\operatorname{Var}(\lambda-\hat Y)/\operatorname{Var}(\lambda)$, where
> $\lambda=\mathbb E[Y\mid\text{stimulus},\text{gaze}]$. It therefore reports
> variance explained in the conditional firing rate rather than in the noisy
> spike counts. The denominator was taken from the Figure 2 estimate over all
> valid bins rather than re-estimated on the scored subset, where the close-pair
> estimator is unusable. Units with a non-positive Figure 2 rate variance were
> excluded rather than clipped.

The corresponding figure label should retain the mathematical definition, such
as “single-trial $r^2/R^2_{\max}$,” because finite-sample values may be negative
or exceed one. That label names the population quantity in (17); the plotted
estimator is (20), and the two coincide up to the sampling ratio in Section 7. A
reference at one denotes the estimated oracle rate variance, not a hard bound on
the plotted estimator.
