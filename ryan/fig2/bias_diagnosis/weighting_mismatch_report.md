# Weighting Mismatch Diagnosis Report

Session: Allen_2022-04-13

```
======================================================================
WEIGHTING MISMATCH DIAGNOSIS
Session: Allen_2022-04-13
======================================================================

======================================================================
STEP 1: Time-bin trial count distribution
======================================================================

  All unique time bins: 107
  n_t distribution (all):
    min=1, max=78, mean=34.3, std=32.1, median=43

  After filtering n_t >= 10: 55 time bins (dropped 52)
  n_t distribution (filtered):
    min=42, max=78, mean=64.3, std=12.2, median=67
    CV(n_t) = 0.190
  n_pairs distribution:
    min=861, max=3003, mean=2111.8, std=747.9
    CV(n_pairs) = 0.354

  Trial counts are NOT uniform across time bins
  Ratio max/min n_t = 1.86
  Ratio max/min n_pairs = 3.49

======================================================================
STEP 2: PSTH covariance under different weightings
======================================================================

  A. Naive estimates (mu_t estimated from all trials in bin):
    Cpsth_uniform (1/T):    0.008384
    Cpsth_trial   (n_t):    0.008776
    Cpsth_paired  (n_pairs): 0.009029

  B. Unbiased estimates (cross-product of distinct trials):
    Cpsth_uniform (1/T):    0.007869
    Cpsth_trial   (n_t):    0.008239
    Cpsth_paired  (n_pairs): 0.008475

  Differences (unbiased):
    Cpsth_paired - Cpsth_uniform:  0.000607
    Cpsth_trial  - Cpsth_uniform:  0.000371
    Cpsth_paired - Cpsth_trial:    0.000236
    Cpsth_mixed  (pair outer, global Erate): 0.011125
    Cpsth_mixed - Cpsth_uniform:   0.003257

  Finite-sample noise in naive estimate:
    Naive - Unbiased (uniform):  0.000516
    Naive - Unbiased (paired):   0.000554
    Naive - Unbiased (trial):    0.000536

  --> Pair-count weighting INFLATES PSTH covariance by 0.000607 (7.7%)

======================================================================
STEP 3: Direct prediction test
======================================================================

  Prediction:
    off_diag(Cpsth_uniform) should match Cpsth from split-half (~0.0082)
    off_diag(Cpsth_paired)  should match Crate_shuff (~0.0107)
    Difference should be ~0.0025

  Observed:
    off_diag(Cpsth_uniform) = 0.007869
    off_diag(Cpsth_paired)  = 0.008475
    Difference              = 0.000607

  Predicted difference / target (0.0025): 0.24x
  --> NOT CONFIRMED: weighting mismatch does not match the observed bias

======================================================================
STEP 4: Verify split-half estimator matches uniform weighting
======================================================================

  off_diag(Cpsth_uniform, analytical):  0.007869
  off_diag(Cpsth_split_half, bagged):   0.008035
  Difference:                           0.000166

  Code analysis of bagged_split_half_psth_covariance:
    - Each time bin contributes 1 row to XA and XB
    - C_k = (XA_c.T @ XB_c) / (n_time - 1)
    - This is a uniform-weight cross-covariance: each time bin
      contributes equally regardless of n_t
    --> CONFIRMED: split-half uses uniform 1/T weighting
  --> Analytical and empirical uniform estimates AGREE (within 2.1%)

======================================================================
STEP 5: Correlation between n_t and PSTH variance
======================================================================

  Per-time-bin PSTH 'variance' = ||mu_t||^2
    Pearson r(n_t, ||mu_t||^2)      = 0.2221  (p = 1.0313e-01)
    Pearson r(n_pairs, ||mu_t||^2)   = 0.2388  (p = 7.9092e-02)
    Spearman r(n_t, ||mu_t||^2)      = 0.1752  (p = 2.0073e-01)

  --> Positive correlation but not significant (p=0.103)
  Saved: weighting_nt_vs_psth_variance.png

======================================================================
STEP 6: Empirical shuffle verification
======================================================================

  Analytical predictions:
    Cpsth_mixed  (pair outer, global Erate):  0.011125
    Cpsth_paired (pair outer, pair Erate):    0.008475
    Cpsth_uniform (1/T outer, uniform Erate): 0.007869

  Running 20 shuffle iterations...

  Results across 20 shuffles:
    Crate_shuff off-diag mean:   0.012120 +/- 0.003964
    Crate_shuff range:           [0.002778, 0.020433]

  Comparison to analytical predictions:
    Crate_shuff - Cpsth_mixed:   0.000994
    Crate_shuff - Cpsth_paired:  0.003644
    Crate_shuff - Cpsth_uniform: 0.004251

  --> Best match: Cpsth_mixed (residual = 0.000994)
  Saved: weighting_shuffle_verification.png

======================================================================
STEP 7: Full quantitative chain
======================================================================

  === QUANTITATIVE CHAIN ===

  1. PSTH covariance (off-diagonal mean, all unbiased):
     Cpsth_uniform  = 0.007869  (1/T weight)
     Cpsth_split    = 0.008035  (empirical split-half)
     Cpsth_paired   = 0.008475  (pair-count weight)
     Cpsth_mixed    = 0.011125  (pair outer, global Erate)
     Crate_shuff    = 0.012120 +/- 0.003964  (empirical)

  2. Covariance-space differences:
     Cpsth_paired - Cpsth_uniform = 0.000607
     Cpsth_mixed  - Cpsth_uniform = 0.003257
     Crate_shuff  - Cpsth_split   = 0.004085
     Crate_shuff  - Cpsth_uniform = 0.004251

  3. Fisher-z conversion:
     fz(CnoiseU_uniform) = 0.083540
     fz(CnoiseU_split)   = 0.083268
     fz(CnoiseC_paired)  = 0.082414
     fz(CnoiseC_mixed)   = 0.075245
     Dz (paired-uniform) = -0.001127
     Dz (mixed-uniform)  = -0.008296

  4. Comparison to observed Dz_shuff ~ -0.006:
     Predicted Dz (paired) = -0.001127
     Predicted Dz (mixed)  = -0.008296
     paired: explains 19% of the observed shift
     mixed: explains 138% of the observed shift

======================================================================
SUMMARY
======================================================================

  1. Trial counts are VARIABLE across time bins (CV = 0.190)
  2. Pair-count vs uniform PSTH cov difference: 0.000607
  3. Correlation r(n_t, ||mu_t||^2) = 0.222 (p = 1.03e-01)
  4. Split-half off-diag = 0.008035
  5. Crate_shuff off-diag = 0.012120
  6. Crate_shuff - split-half = 0.004085
  7. Predicted Dz from weighting: paired=-0.001127, mixed=-0.008296
  8. Observed Dz_shuff: ~-0.006

  VERDICT: Weighting mismatch is a MAJOR contributor to the shuffle null shift

  KEY FINDING:
    The empirical Crate_shuff (0.012120) exceeds
    the split-half Cpsth (0.008035) by 0.004085.
    However, this excess is larger than what pair-count weighting alone
    predicts. The intercept fitting (extrapolation from distance bins)
    adds additional upward bias under the null because shuffled eyes
    create a non-flat distance curve that the linear fit extrapolates.
```
