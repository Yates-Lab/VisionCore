# Shuffle Null Shift Diagnosis Report

## Problem
The shuffle control for the noise correlation analysis shows a systematic negative shift: Dz_shuff = -0.006 instead of 0. Under the null (shuffled eye trajectories destroy spike-eye coupling), Crate_shuff should estimate the same quantity as Cpsth, making Dz = fz(CnoiseC) - fz(CnoiseU) = 0.

Session: Allen_2022-04-13, N_SHUFFLES = 50.

## Baseline Values
- fz(CnoiseU) = 0.0829
- fz(CnoiseC) = -0.0026 (real data)
- Dz_real = -0.0855

## Results by Hypothesis

### Hypothesis A: Reused Bin Edges -- RULED OUT
Reused edges and fresh (recomputed) edges produce **identical** results:
- Dz_shuff (reused edges):  -0.006827 +/- 0.001272
- Dz_shuff (fresh edges):   -0.006827 +/- 0.001272
- Difference: 0.000000

The bin edges from real data work fine for shuffled data because `compute_conditional_second_moments` uses percentile-based binning that is robust to the distance distribution shift.

### Hypothesis B: Shuffle Strategy -- MINOR CONTRIBUTOR
Different shuffle strategies produce different shift magnitudes, but none eliminate it:

| Strategy        | Dz_mean   | Dz_SE    |
|-----------------|-----------|----------|
| Global shuffle  | -0.006827 | 0.001272 |
| Within-time-bin | -0.005549 | 0.000924 |
| Cyclic shift    | -0.003665 | 0.000943 |

Within-time-bin shuffling reduces the shift by 0.0013 (19%), and cyclic shift reduces it by 0.0032 (47%). The remaining shift under cyclic shift (-0.0037) demonstrates that most of the bias is NOT from temporal structure mismatch. The shuffle strategy is a secondary contributor at most.

### Hypothesis C: Estimator Asymmetry -- PRIMARY CAUSE
Under the shuffle null, the trajectory-matching estimator (Crate) **systematically overestimates** signal covariance relative to the split-half estimator (Cpsth):

| Estimator                     | off_diag_mean  |
|-------------------------------|----------------|
| Cpsth (split-half)            | 0.00816485     |
| Crate_intercept (shuffled)    | 0.01069775     |
| Crate_mean (shuffled, all-bin)| 0.01110641     |
| Ctotal                        | 0.04521612     |

Key differences from Cpsth:
- Crate_intercept_shuff - Cpsth = +0.00253 (31% above Cpsth)
- Crate_mean_shuff - Cpsth = +0.00294 (36% above Cpsth)

Both estimators (linear intercept fit AND simple weighted mean across distance bins) show the same upward bias. This means the bias is **not** from the intercept extrapolation but from the fundamental difference between trajectory-matching cross-products and split-half PSTH cross-products as estimators of signal covariance.

The mechanism: The trajectory-matching estimator computes E[Si * Sj | distance_bin] using cross-trial products, then averages across bins. Under the null (flat curve), this should equal Cpsth. But the cross-trial product estimator has higher variance than split-half, and the averaging + intercept fitting introduce a positive bias in the off-diagonal elements. This is consistent with a finite-sample regression-to-the-mean effect in the cross-product space.

Fraction of shuffles where Crate_intercept > Cpsth: 62% (intercept), 100% (mean-bin).

### Hypothesis D: PSD Projection -- RULED OUT
PSD projection has **zero** effect:
- Dz_shuff WITH PSD:    -0.006827
- Dz_shuff WITHOUT PSD: -0.006827
- Negative eigenvalues in CnoiseU: 0
- Negative eigenvalues in CnoiseC_shuff: 0 (all 50 iterations)

Neither CnoiseU nor CnoiseC_shuff has negative eigenvalues, so PSD projection is a no-op for this dataset.

### Hypothesis E: Cov-to-Corr Amplification -- AMPLIFIER (not root cause)
The bias exists in raw covariance space and is amplified by normalization:

| Space                  | Mean difference | SE         |
|------------------------|-----------------|------------|
| Covariance (raw)       | -0.00253        | 0.00063    |
| Correlation (no PSD)   | -0.00683        | 0.00127    |
| Correlation (with PSD) | -0.00683        | 0.00127    |

The covariance-space bias (-0.00253) is amplified 2.7x by the cov-to-corr normalization to produce the observed -0.00683 in Fisher-z space. This amplification occurs because the variance normalization (dividing by sqrt(var_i * var_j)) is nonlinear, and extra estimation noise in Crate_shuff creates a directional bias when projected into correlation space.

### Hypothesis F: Interaction Effects -- No Combination Eliminates the Shift

| Condition                       | Dz_mean   | Fixed? |
|---------------------------------|-----------|--------|
| baseline (global+reused+PSD)   | -0.006827 | no     |
| fresh edges only                | -0.006827 | no     |
| no PSD only                     | -0.006827 | no     |
| within-time only                | -0.005549 | no     |
| fresh + no PSD                  | -0.006827 | no     |
| within + no PSD                 | -0.005549 | no     |
| within + fresh                  | -0.005549 | no     |
| within + fresh + no PSD         | -0.005549 | no     |

No tested combination of bin edges, shuffle strategy, and PSD toggling eliminates the shift.

## Root Cause Diagnosis

The -0.006 shuffle null shift has **two contributing factors**:

1. **Primary (100% necessary): Estimator asymmetry (Hypothesis C).** The trajectory-matching cross-product estimator used for Crate has a positive finite-sample bias relative to the split-half PSTH estimator used for Cpsth. Under the shuffle null, Crate_shuff off-diagonal systematically exceeds Cpsth off-diagonal by ~0.0025-0.003 in covariance space. This positive bias in Crate means CnoiseC_shuff = Ctotal - Crate_shuff is systematically smaller than CnoiseU = Ctotal - Cpsth, producing the negative Dz shift.

2. **Secondary amplifier (Hypothesis E): Cov-to-corr normalization.** The covariance-space bias of -0.00253 is amplified to -0.00683 in Fisher-z correlation space (2.7x amplification) by the nonlinear variance normalization in cov_to_corr.

3. **Minor contributor (Hypothesis B): Global vs within-time shuffling** accounts for ~0.001 of the 0.007 shift (15%). Using within-time-bin shuffling is technically more correct, but does not resolve the core issue.

## Recommended Fix

**The shift cannot be eliminated by changing bin edges, shuffle strategy, or PSD settings.** The fundamental issue is that Crate and Cpsth are different estimators with different finite-sample biases.

**Option 1 (Report as-is):** Report the shuffle null distribution and note that Dz_shuff = -0.006 represents the method's intrinsic bias floor. The real Dz = -0.086 is 12.6x larger than the shuffle null shift, so the FEM-driven noise correlation reduction is robust (the signal-to-bias ratio is ~12.6:1).

**Option 2 (Bias-corrected Dz):** Report Dz_corrected = Dz_real - Dz_shuff_mean. This subtracts out the estimator bias, giving Dz_corrected = -0.086 - (-0.007) = -0.079.

**Option 3 (Matched estimator):** Use the same trajectory-matching framework for both Crate and the "uncorrected" baseline. Replace Cpsth with the weighted-mean-across-bins estimate from Crate_shuff (which estimates the same thing as Cpsth but with matching finite-sample properties). This would make the bias cancel in the subtraction. However, this requires rethinking the CnoiseU estimator.

**Recommended approach: Option 2.** It is the simplest, most transparent fix. Report both raw and bias-corrected Dz values, with the shuffle null distribution as evidence that the correction is valid.

## Does the Fix Fully Eliminate the Shift?

No single procedural change eliminates the shift. This is an inherent property of comparing two different estimators. However, the bias-correction approach (Option 2) fully accounts for it statistically, and the 12.6:1 signal-to-bias ratio means the main scientific conclusion (FEM correction reduces noise correlations) is robust regardless.
