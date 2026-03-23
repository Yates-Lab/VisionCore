# H3 + H5 Report: PSD Projection Asymmetry & Jensen's Inequality

**Session:** Allen_2022-04-13 | **Neurons:** 49 (after filtering) | **Windows:** 3667

---

## Part 1: Eigenspectra

| Metric | CnoiseU | CnoiseC |
|--------|---------|---------|
| Negative eigenvalues | 0/49 | 1/49 |
| Total negative eigenvalue mass | 0.000 | -1.542 |
| Negative mass fraction | 0.000 | -0.088 |
| Frobenius norm in neg eigenspace | 0.000 | 0.490 |
| Min eigenvalue | +0.114 | -1.542 |
| Max eigenvalue | +3.607 | +0.941 |

CnoiseU is already PSD (no negative eigenvalues). CnoiseC has exactly one large negative eigenvalue (-1.54), which carries ~9% of the total eigenvalue mass and ~49% of the Frobenius norm in the negative eigenspace. This confirms CnoiseC is substantially more indefinite than CnoiseU.

## Part 2: PSD Projection Effect

| Metric | CnoiseU | CnoiseC |
|--------|---------|---------|
| Mean corr (no PSD) | +0.0824 | -0.0588 |
| Mean corr (PSD) | +0.0824 | -0.0026 |
| Delta z (PSD effect) | 0.0000 | +0.0694 |

**PSD has zero effect on CnoiseU** (it is already PSD) but **raises CnoiseC correlations by +0.069 in Fisher z**. This is an asymmetric effect: PSD pulls corrected noise correlations up toward zero, reducing the apparent negative bias.

Asymmetric PSD bias = +0.069 (PSD helps corrected more than uncorrected).

## Part 3: Full Decomposition

| Condition | Dz (corrected - uncorrected) |
|-----------|------------------------------|
| Both PSD (paper pipeline) | **-0.086** |
| Neither PSD | **-0.155** |
| PSD on U only | -0.155 |
| PSD on C only | -0.086 |

**Key finding:** The bias is -0.155 without any PSD projection. PSD projection partially *masks* the bias by raising CnoiseC correlations, reducing the observed Dz from -0.155 to -0.086.

PSD contribution = +0.069 (reduces magnitude of Dz by 45%).

**PSD projection is not the cause of the negative bias. It actually attenuates it.** The true underlying Dz is even more negative (-0.155) than what the paper pipeline reports (-0.086).

## Part 4: Jensen's Inequality

| Metric | CnoiseU | CnoiseC |
|--------|---------|---------|
| Diagonal CV (bootstrap) | 0.045 | 0.045 |
| z (standard) | +0.083 | -0.003 |
| z (clipped diag) | +0.087 | -0.003 |

| Normalization | Dz |
|---------------|-----|
| Standard | -0.086 |
| Clipped diag | -0.091 |
| Median diag | -0.158 |

Jensen contribution to Dz: +0.005 (5.8% of Dz_psd). The diagonal coefficient of variation is only ~4.5%, so normalization noise is small. **H5 is negligible.**

## Part 5: Monte Carlo

Simulating random matrices with 40 neurons and varying negative eigenvalue mass:

- At the real CnoiseC negative mass fraction (0.088), the Monte Carlo predicted PSD shift is only +0.0007 in Fisher z.
- This is much smaller than the observed +0.069 shift, meaning the real data's PSD effect is dominated by the **structured** nature of the single large negative eigenvalue, not generic indefiniteness.
- The MC uses random (unstructured) negative eigenvalues, which average out; the real data has a single dominant negative eigenvalue that systematically shifts all off-diagonal correlations.

## Part 6: Bottom Line

| Mechanism | Contribution to Dz | % of observed Dz (-0.086) |
|-----------|--------------------:|-------------------------:|
| H3: PSD projection | +0.069 (attenuates) | -81% (reduces bias) |
| H5: Jensen's inequality | +0.005 | -6% (reduces bias) |
| **Underlying covariance bias** | **-0.155** | **181%** |

### Verdict

**H3 (PSD projection) does NOT cause the negative bias -- it partially masks it.** Without PSD, the bias would be -0.155 instead of -0.086. PSD projection raises CnoiseC correlations (because it clamps the single large negative eigenvalue to 0), reducing the apparent bias by ~45%.

**H5 (Jensen's inequality) is negligible** (5% contribution), as diagonal variance estimates are quite stable (CV ~4.5%).

**The root cause is upstream:** Crate off-diagonals exceed Ctotal off-diagonals, making CnoiseC = Ctotal - Crate have genuinely negative off-diagonal covariances. This negative covariance structure is what produces the -0.155 underlying Dz. PSD and normalization are second-order effects that partially attenuate it.

### Implication for the remaining hypotheses

The ~-0.155 "raw" Dz (no PSD) needs to be explained by:
- Why Crate off-diags > Ctotal off-diags (the fundamental data fact)
- H6 (shuffle null bias) may help calibrate what portion is expected under the null
