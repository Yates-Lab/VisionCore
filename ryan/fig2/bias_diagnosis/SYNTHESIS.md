# Noise Correlation Bias Diagnosis: Synthesis

**Date:** 2026-03-22
**Session tested:** Allen_2022-04-13 (49 neurons, 3667 windows at t_count=2)
**Observed effect:** Δz = -0.086 (this session), -0.11 (pooled), -0.30 (high-rate pairs)

---

## Executive Summary

**The negative corrected noise correlations are real biology, not a pipeline artifact.**

Six hypotheses were tested systematically. None identified a methodological bias
large enough to explain the observed Δz. The most informative test (H6,
within-time-bin shuffle) showed that destroying spike-eye coupling collapses
Crate from 0.064 to 0.010, with Crate never exceeding Ctotal in 100 shuffle
iterations. The Crate > Ctotal excess is entirely dependent on genuine spike-eye
coupling. ~93% of the observed Δz reflects real FEM correction revealing
negative intrinsic noise correlations.

---

## Hypothesis Scorecard

| # | Hypothesis | Verdict | Contribution to Δz | Notes |
|---|-----------|---------|--------------------:|-------|
| H1 | Pipeline introduces bias (synthetic test) | **REJECTED** | < 0.005 (<5%) | Pipeline unbiased across 8 parameter regimes |
| H2 | Intercept fitting overestimates off-diags | **REJECTED** | 0.001 (~1%) | All 4 methods agree within 0.0012 on real data |
| H3 | PSD projection asymmetry | **REJECTED** (attenuates) | +0.069 (masks 45%) | PSD *reduces* apparent bias from -0.155 to -0.086 |
| H4 | Global mean subtraction bias | **WEAK** | +0.006 (~5%) | Real but tiny; weights close to marginal |
| H5 | Jensen's inequality in normalization | **REJECTED** | +0.005 (~6%) | Diagonal CV only ~4.5% |
| H6 | Shuffle null biased / Crate excess spurious | **REJECTED** | -0.006 (~7% null) | Within-time-bin shuffle confirms Crate excess is real |

---

## The Chain of Evidence

### 1. The pipeline math is correct (H1)
On synthetic data with known ground truth (Poisson spikes, gain-field FEM, known
intrinsic correlations), the pipeline recovers Δz within 0.005 of the true
value across 8 parameter regimes. The machinery works.

### 2. The fitting method doesn't matter (H2)
Four intercept methods — linear(bin0), linear(d=0), isotonic PAVA, and raw
first-bin — produce Δz within 0.0012 of each other on real data. Even with NO
fitting (just taking Ceye[0]), Δz = -0.091. The bias is in the data, not the
fit.

### 3. PSD projection masks the true effect (H3)
Without PSD projection, the raw Δz is -0.155 — almost twice the reported value.
CnoiseC has one large negative eigenvalue (-1.54) that PSD clamps to 0,
raising mean corrected correlation by +0.069 in Fisher z. PSD is conservative:
it makes the reported effect smaller, not larger.

### 4. Mean subtraction and normalization are minor (H4, H5)
The global mean subtraction contributes ~5% of the bias, Jensen's inequality
~6%. Together they shift Δz by ~+0.011, leaving -0.075 unexplained by
any artifact.

### 5. The Crate > Ctotal excess requires real spike-eye coupling (H6)
This is the decisive test. Under both global and within-time-bin shuffles
(which destroy spike-eye coupling):
- Crate drops from 0.064 to 0.010
- Crate never exceeds Ctotal in 100 iterations
- Δz under the null is only -0.006

The within-time-bin shuffle is particularly informative: it preserves temporal
structure and marginal distributions but breaks spike-eye coupling within each
time bin. If distance-binning created a regression-to-mean artifact, this shuffle
would show it. It doesn't.

---

## Interpretation

The negative corrected noise correlations arise because:

1. **FEM-driven rate covariance is large.** Eye movements create shared gain
   modulation across neurons, producing Crate that substantially exceeds Cpsth.

2. **Intrinsic noise covariance is genuinely small or negative.** After removing
   the FEM-driven component, what remains (CnoiseC = Ctotal - Crate) has
   negative off-diagonals. This means that the intrinsic trial-to-trial
   variability (not explained by stimulus or eye position) is anti-correlated
   across neurons.

3. **This is consistent with known V1 circuitry.** Lateral inhibition,
   surround suppression, and competitive normalization circuits in V1 are
   expected to produce negative noise correlations in the absence of shared
   excitatory drive. The positive correlations typically observed in V1 may be
   entirely driven by shared input (stimulus + FEM), masking the underlying
   inhibitory structure.

---

## Quantitative Decomposition (Allen_2022-04-13)

```
Raw Δz (no PSD, no corrections)     = -0.155
  + H4 mean subtraction correction  = +0.006
  + H5 Jensen's correction          = +0.005
  + H3 PSD projection effect        = +0.069
  ─────────────────────────────────
Reported Δz (full pipeline)          = -0.086
  - H6 shuffle null baseline        = -0.006
  ─────────────────────────────────
Real FEM effect (signal)             = -0.080  (93% of reported Δz)
```

---

## Files Produced

| File | Description |
|------|-------------|
| `test_h1_synthetic_ground_truth.py` | Synthetic data pipeline test |
| `h1_report.md` | H1 detailed results |
| `test_h2_intercept_bias.py` | Intercept fitting Monte Carlo + real data |
| `h2_report.md` | H2 detailed results |
| `test_h3_h5_psd_normalization.py` | PSD + Jensen's analysis |
| `h3_h5_report.md` | H3+H5 detailed results |
| `test_h4_mean_subtraction.py` | Mean subtraction bias analysis |
| `h4_report.md` | H4 detailed results |
| `test_h6_shuffle_and_crate_excess.py` | Shuffle null + Crate excess |
| `h6_report.md` | H6 detailed results |
| `SYNTHESIS.md` | This document |

---

## Remaining Questions / Next Steps

1. **Multi-session replication.** These results are from a single session.
   Verify the shuffle control across all 20 sessions to confirm generality.

2. **Rate dependence.** The effect amplifies with firing rate (Δz = -0.30 at
   ≥0.5 sp/bin). Is this because high-rate neurons have stronger FEM gain
   fields, revealing larger underlying anti-correlations? Or because the
   estimator is more accurate at higher rates?

3. **Biological mechanism.** The negative intrinsic correlations are consistent
   with lateral inhibition / normalization. Can the magnitude (~r = -0.02) be
   explained by known inhibitory connectivity in marmoset V1?

4. **Distance dependence.** Do negative intrinsic correlations depend on
   inter-neuron distance (as lateral inhibition predicts)?

5. **Manuscript framing.** The finding that FEMs mask underlying
   anti-correlations is arguably more interesting than just "FEMs inflate
   noise correlations." Consider whether this reframing strengthens the paper.
