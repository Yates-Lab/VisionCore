# Noise Correlation Bias Analysis: Delta-z Shuffle Null

**Date:** 2026-03-22
**Script:** `explore_noise_corr_bias.py`
**Session used for exploration:** Allen_2022-04-13
**Status:** Open — root cause not yet identified

## Problem Statement

The shuffle null distribution for delta-z (the change in mean Fisher-z noise
correlation after FEM correction) has a consistent **negative bias**. That is,
even when the eye-spike association is destroyed by permutation, the analysis
reports a small reduction in noise correlations. This is problematic because
the real effect is also negative — FEM correction reduces noise correlations —
so the bias is in the same direction as the effect of interest.

Observed shuffle bias (single session, linear intercept, eval at first bin):

| Window (ms) | Observed delta-z | Shuffle mean delta-z | Bias as % of effect |
|-------------|-----------------|---------------------|---------------------|
| 8.3         | -0.090          | -0.009              | ~10%                |
| 16.7        | -0.137          | -0.004              | ~3%                 |
| 33.3        | -0.195          | -0.002              | ~1%                 |
| 66.7        | -0.161          | -0.055              | ~34%                |


## Hypotheses Tested (All Rejected)

### Hypothesis 1 (rejected): PSD Projection Asymmetry

**Rationale:** Both CnoiseU and CnoiseC are projected to the nearest positive
semi-definite matrix before converting to correlations. If CnoiseC has more
negative eigenvalues than CnoiseU (due to noisy element-wise Crate estimation),
PSD projection would clip more aggressively, inflating the diagonal
proportionally more than the off-diagonals, and thus reducing correlations
after normalization.

**Test:** Computed delta-z with and without `project_to_psd` for both real
data and every shuffle iteration. Counted negative eigenvalues in each case.

**Result:** The shuffle null produces **zero negative eigenvalues** in
CnoiseC_shuff. PSD projection is a complete no-op for shuffled data. The
shuffle delta-z is identical with and without PSD projection.

For the real data, PSD projection actually **reduces** the apparent effect
(shifts delta-z toward zero), because CnoiseC_real does have 1 negative
eigenvalue. Clipping it to zero adds positive structure back, inflating
corrected correlations. PSD projection is therefore conservative — it makes
the real effect appear smaller, not larger.

| Window (ms) | delta-z (with PSD) | delta-z (no PSD) | PSD shift |
|-------------|-------------------|-----------------|-----------|
| 8.3         | -0.090            | -0.183          | +0.093    |
| 16.7        | -0.137            | -0.324          | +0.186    |
| 33.3        | -0.195            | -0.383          | +0.188    |
| 66.7        | -0.161            | -0.398          | +0.237    |

**Conclusion:** PSD projection is not the source of the shuffle bias. It is
conservative for the real effect and irrelevant for the shuffle null.


### Hypothesis 2 (rejected): Jensen's Inequality through cov_to_corr

**Rationale:** Converting covariance to correlation involves dividing the
off-diagonal by the square root of the product of two diagonal elements. This
is a nonlinear operation. When the covariance matrix has zero-mean noise added
(as it does under the shuffle null, where Crate_shuff = Cpsth + noise), the
expected correlation might not equal the correlation of the expected covariance.

**Test:** Minimal Monte Carlo simulation (Step 15 in explore script). Created
a known noise covariance matrix, drew samples, computed sample covariance, then
added zero-mean symmetric noise at varying scales and computed delta-z in both
correlation and covariance space.

**Result:** Adding zero-mean noise to a covariance matrix and converting to
correlations does **not** produce a negative bias. The simulation showed
delta-z centered at zero (or very slightly positive) for all noise scales
tested (0.01 to 0.2). The Jensen's inequality effect through cov_to_corr
normalization is negligible and goes in the wrong direction.

| Noise scale | Mean delta-z | 95% CI |
|-------------|-------------|--------|
| 0.01        | +0.00001    | [-0.00066, +0.00072] |
| 0.05        | +0.00008    | [-0.00322, +0.00356] |
| 0.20        | +0.00320    | [-0.01168, +0.01971] |

**Conclusion:** The nonlinear cov-to-corr step does not produce the observed
negative bias. A more structured noise model (matching the actual regression
error correlations) might behave differently, but the generic mechanism does
not explain the finding.


### Hypothesis 3 (rejected): Weighting Mismatch Between Crate and Cpsth

**Rationale:** The split-half Cpsth estimator gives each time bin **equal
weight** (one row per time bin in XA, XB). The cross-product estimator in
`compute_conditional_second_moments` weights each time bin by pair count
(proportional to n_t^2). If n_t correlates with PSTH signal strength (||mu_t||),
Crate_shuff would systematically overestimate the rate covariance relative to
Cpsth, biasing CnoiseC negative and thus delta-z negative.

**Test 1 (real data, Step 15):** Computed pair-count-weighted Cpsth and used it
as the baseline instead of uniform-weighted Cpsth. Ran shuffle null with both
baselines.

**Test 2 (simulation, Step 16):** Monte Carlo with two conditions: (a) n_t
correlated with ||mu_t|| (time bins with more windows have stronger PSTH), and
(b) n_t independent of ||mu_t||. Compared standard (uniform Cpsth) vs matched
(pair-count-weighted Cpsth) baselines.

**Result:** The bias persists regardless of weighting correction. In the
simulation, both the correlated and uncorrelated conditions show a small
negative bias (~-0.01 to -0.02), and the matched-weight baseline does not
eliminate it.

| Condition    | Standard baseline | Matched baseline |
|--------------|------------------|-----------------|
| Correlated   | -0.0149          | -0.0208         |
| Uncorrelated | -0.0084          | -0.0083         |

**Conclusion:** The weighting mismatch is not the primary source of the bias.
There may be a contribution when n_t and ||mu_t|| are correlated, but
correcting the weights does not eliminate the bias. The simulation itself
shows a residual bias that must have another source.


## What We Do Know

1. **The bias is real and consistent** across counting windows, though
   non-monotonic in magnitude.

2. **PSD projection is irrelevant** to the shuffle null (no negative
   eigenvalues) but is conservative for the real effect.

3. **The bias is small** for short counting windows (3-10% of the effect at
   8.3 and 16.7 ms) but substantial at the longest window (34% at 66.7 ms).

4. **P-values remain valid** because the shuffle test compares observed delta-z
   against the empirical null distribution. Whatever the bias source, both
   the real and null distributions are subject to the same pipeline, so the
   relative comparison is fair.

5. **The bias is conservative** for significance testing — the shuffle null
   is shifted negative, making it harder (not easier) to reject.


## Open Questions

- What is the actual mechanism? The bias appears in a minimal simulation
  (Step 16) even without the specific noise structure of the real data,
  suggesting it is inherent to the cross-product-based Crate estimation
  combined with correlation normalization. But the specific pathway has not
  been isolated.

- Does the bias scale with a specific parameter of the analysis
  (number of distance bins, regression method, number of neurons, number
  of time bins)? Systematic parameter sweeps may help isolate the mechanism.

- Is there a closed-form expression for the expected bias? This would enable
  analytical correction rather than relying on shuffle subtraction.


## Plan Moving Forward

### For Figure 2 (panels F, G, H — noise correlations)

1. **Keep the current analysis pipeline unchanged.** The p-values from the
   shuffle test are valid because both real and null pass through the same
   pipeline.

2. **Report effect sizes at short counting windows** (8.3 ms and 16.7 ms)
   where the bias is smallest (3-10% of the effect).

3. **Add a methods note** stating that significance is assessed against the
   empirical shuffle distribution, which accounts for any procedural biases
   in the delta-z statistic.

4. **Keep PSD projection.** It is conservative (reduces the apparent effect
   size) and is needed for downstream subspace analyses. Its effect on the
   shuffle null is zero.

### For continued investigation

5. **Parameter sweeps** on the number of distance bins, regression method
   (linear vs PAVA vs raw), and minimum trials per time bin to characterize
   how the bias scales.

6. **Analytical derivation** of the expected cross-product covariance under
   the null, accounting for within-time-bin structure, to identify any
   systematic difference from the split-half Cpsth estimator.

7. **Alternative test statistics** that might not exhibit the bias (e.g.,
   working in covariance space, using a different normalization, or using
   a ratio-based statistic that avoids per-pair correlation computation).


## Key Code Locations

| What | File | Lines |
|------|------|-------|
| PSD projection | `VisionCore/subspace.py` | 5-55 |
| Cov to correlation | `VisionCore/covariance.py` | 18-58 |
| Conditional second moments | `VisionCore/covariance.py` | 251-341 |
| Linear intercept fit | `VisionCore/covariance.py` | 396-478 |
| PAVA intercept fit | `VisionCore/covariance.py` | 359-393 |
| Delta-z computation | `ryan/fig2/generate_figure2.py` | 317-323 |
| Shuffle loop | `ryan/fig2/generate_figure2.py` | 335-355 |
| PSD diagnostic | `ryan/fig2/explore_noise_corr_bias.py` | Step 14 |
| Weighting diagnostic | `ryan/fig2/explore_noise_corr_bias.py` | Step 15 |
| Weighting simulation | `ryan/fig2/explore_noise_corr_bias.py` | Step 16 |
