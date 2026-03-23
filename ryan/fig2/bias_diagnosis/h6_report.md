# H6 Report: Shuffle Null Bias and Crate > Ctotal Excess

## Summary

The Crate > Ctotal excess is **real and driven by genuine spike-eye coupling**, not a methodological artifact. Under both global and within-time-bin shuffles (which destroy spike-eye coupling), Crate_shuff falls *well below* Ctotal, never above it. In Fisher-z space, shuffle bias accounts for only ~7% of the observed Dz = -0.086. The remaining ~93% reflects real FEM-driven rate covariance that, when subtracted, reveals negative intrinsic noise correlations.

## Baseline (real data)

| Metric | Value |
|---|---|
| fz(CnoiseU) | 0.0829 |
| fz(CnoiseC) | -0.0026 |
| Dz = fz(C) - fz(U) | -0.0855 |
| off-diag mean Ctotal | 0.0452 |
| off-diag mean Crate | 0.0644 |
| off-diag mean CnoiseC | -0.0192 |

## Part 1 — Global shuffle null (50 iterations)

Permuting eye trajectories across all trials destroys temporal structure and any spike-eye coupling.

- Crate_shuff off-diag: 0.0107 +/- 0.0044
- Crate_shuff - Ctotal: -0.0345
- Fraction of shuffles with Crate_shuff > Ctotal: **0/50 (0%)**
- fz(CnoiseC_shuff): 0.0761 +/- 0.0090

**Result**: Under the global null, Crate is far *below* Ctotal. The shuffle null is not biased toward inflating Crate. The real Crate excess over Ctotal requires real spike-eye coupling.

## Part 2 — Direct Crate excess decomposition

| Component | Off-diag covariance |
|---|---|
| Crate_real | 0.0644 |
| Crate_shuff (global) | 0.0107 |
| Ctotal | 0.0452 |
| Crate_real - Ctotal (total excess) | +0.0192 |
| Crate_shuff - Ctotal (null excess) | -0.0345 |
| Crate_real - Crate_shuff (FEM-driven) | +0.0537 |

The FEM-driven component (0.054) is much larger than the total excess (0.019), meaning real eye-modulation raises Crate far above the shuffle baseline. The null baseline itself sits below Ctotal.

## Part 3 — Within-time-bin shuffle (regression-to-mean test)

This is the critical test. Within each time bin, spike-count vectors are randomly reassigned to eye trajectories. This preserves temporal structure and marginal distributions but breaks spike-eye coupling *within* each time bin. If the distance-binning procedure itself creates a finite-sample regression-to-mean bias, this shuffle would show Crate > Ctotal.

- Crate_within off-diag: 0.0104 +/- 0.0033
- Crate_within - Ctotal: -0.0348
- Fraction with Crate_within > Ctotal: **0/50 (0%)**
- fz(CnoiseC_within): 0.0770 +/- 0.0066

**Result**: No regression-to-mean bias from distance binning. With no spike-eye coupling, Crate collapses to ~0.010 regardless of whether the shuffle is global or within-time-bin. The distance-conditioning procedure does not spuriously inflate Crate.

## Part 4 — Comparison across conditions

| Condition | Crate off-diag | fz(CnoiseC) | Crate - Ctotal |
|---|---|---|---|
| Real data | 0.0644 | -0.0026 | +0.0192 |
| Global shuffle | 0.0107 | 0.0761 | -0.0345 |
| Within-time shuffle | 0.0104 | 0.0770 | -0.0348 |

The two shuffle conditions are nearly identical, confirming temporal structure is not relevant — what matters is the spike-eye coupling.

Dz values:
- Real: -0.0855
- Global shuffle: -0.0068
- Within-time shuffle: -0.0060

## Part 5 — Quantification

### Fisher-z decomposition (the meaningful metric)

| Component | Dz |
|---|---|
| Total (real) | -0.0855 |
| Shuffle null (within-bin) | -0.0060 |
| Real FEM signal | -0.0796 |
| **Bias fraction** | **7.0%** |
| **Real signal fraction** | **93.0%** |

The small -0.006 Dz from the shuffle null likely reflects residual PSD projection asymmetry (as established in H3) rather than any distance-binning artifact.

## Conclusions

1. **The shuffle null is NOT biased toward negative Dz.** Both global and within-time-bin shuffles show Crate well below Ctotal (not above it). The ~0.006 residual Dz in the null is negligible.

2. **No regression-to-mean bias from distance binning.** The within-time-bin shuffle (the precise null for this concern) shows no Crate inflation. The distance-conditioning procedure is unbiased.

3. **The Crate > Ctotal excess is entirely driven by real spike-eye coupling.** When coupling is destroyed by shuffling, Crate drops from 0.064 to 0.010. The 0.054 FEM-driven component is the real signal.

4. **Negative intrinsic noise correlations appear to be real biology.** CnoiseC = Ctotal - Crate has genuinely negative off-diagonals because eye-driven rate covariance (Crate) genuinely exceeds total covariance (Ctotal). This is consistent with lateral inhibition or competitive interactions producing anti-correlated intrinsic noise that is normally masked by positive FEM-driven correlations.

5. **~93% of Dz = -0.086 reflects real FEM correction.** Only ~7% is attributable to any residual methodological bias (likely the PSD projection asymmetry from H3).

## Figures

- `h6_shuffle_and_crate_excess.png` — Main 4-panel summary figure
- `h6_shuffle_scatter.png` — Supplementary scatter and distribution plots
