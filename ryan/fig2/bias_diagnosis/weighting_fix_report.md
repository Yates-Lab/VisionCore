# Weighting Fix Test Report

Session: Allen_2022-04-13
Global shuffles: 50
Within-bin shuffles: 20

```
======================================================================
WEIGHTING FIX TEST
Pair-count-weighted mean subtraction for noise correlation bias
======================================================================

Computing baseline covariances...

======================================================================
PART 2: Real data — original vs fixed pipeline
======================================================================

  fz(CnoiseU)                = 0.082288
  fz(CnoiseC_orig)           = 0.000367
  fz(CnoiseC_fixed)          = 0.000390
  Dz_real (original)         = -0.081920
  Dz_real (fixed)            = -0.081897
  Change in Dz_real          = 0.000023
  off_diag_mean(Crate_orig)  = 0.06176843
  off_diag_mean(Crate_fixed) = 0.06169379
  off_diag_mean(Cpsth)       = 0.00851813

  Erate difference (max abs)  = 0.00051148
  Erate difference (mean abs) = 0.00010702

======================================================================
PART 3: Global shuffle null (50 iterations)
======================================================================

  ORIGINAL pipeline:
    Mean Dz_shuff  = 0.000979 +/- 0.001253 (SE)
    Std  Dz_shuff  = 0.008862
    Range          = [-0.024912, 0.016907]

  FIXED pipeline (Approach A: pair-weighted mean):
    Mean Dz_shuff  = 0.001125 +/- 0.001253 (SE)
    Std  Dz_shuff  = 0.008858
    Range          = [-0.024757, 0.017047]

  Shift reduction  = -0.000146
  Fixed null centered at 0? True

  t-test vs 0 (original):  t=0.773, p=0.4431
  t-test vs 0 (fixed):     t=0.889, p=0.3785

======================================================================
PART 4: Within-time-bin shuffle (20 iterations)
======================================================================

  ORIGINAL pipeline:
    Mean Dz_shuff  = 0.002766 +/- 0.001408 (SE)

  FIXED pipeline:
    Mean Dz_shuff  = 0.002911 +/- 0.001407 (SE)

  Shift reduction  = -0.000145

======================================================================
PART 5: Signal preservation check
======================================================================

  Dz_real (original) = -0.081920
  Dz_real (fixed)    = -0.081897
  Change             = 0.000023
  |Change| < 0.005?  YES
  --> Fix preserves the real-data signal (change < 0.005)

======================================================================
PART 6: Summary
======================================================================

                                Original        Fixed       Change
  ------------------------- ------------ ------------ ------------
  Real data Dz                 -0.081920    -0.081897    +0.000023
  Shuffle null Dz               0.000979     0.001125    +0.000146
  Shuffle null SE               0.001253     0.001253             
  |Signal/Null|                     83.7x         72.8x             

  Within-time-bin shuffle:
                                Original        Fixed       Change
  ------------------------- ------------ ------------ ------------
  Shuffle null Dz               0.002766     0.002911    +0.000145

  Global shuffle t-test vs 0:
    Original: p = 0.4431
    Fixed:    p = 0.3785
  Fixed shuffle null is NOT significantly different from 0 (p=0.379)

  Figure saved: /home/ryanress/v1-fovea/VisionCore/ryan/fig2/bias_diagnosis/weighting_fix_results.png
```
