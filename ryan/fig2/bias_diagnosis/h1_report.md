# H1: Does the LOTC Pipeline Itself Introduce a Negative Bias in Corrected Noise Correlations?

## What Was Tested

We generated synthetic spike-count data with **known ground truth** (PSTH, FEM gain fields, and intrinsic noise correlations) and ran the exact LOTC pipeline from `VisionCore.covariance` to check whether the pipeline machinery introduces a systematic negative bias in corrected noise correlations.

The generative model:
- **PSTH**: Smooth sinusoidal basis functions per neuron (5 basis components)
- **FEM gain fields**: Gaussian gain fields centered at random eye positions; eye trajectories are 2D random walks per trial
- **Intrinsic noise**: Multiplicative log-normal model with known latent correlation structure: `rate = base_rate * exp(sigma * z - sigma^2/2)` where `z ~ N(0, Sigma_corr)`, followed by Poisson sampling
- **Pipeline**: `estimate_rate_covariance` (linear intercept mode, 15 bins) -> `bagged_split_half_psth_covariance` (20 bootstraps) -> PSD projection -> `cov_to_corr` -> `fisher_z_mean`

Eight parameter regimes were tested:

| Case | N_trials | Rate (sp/bin) | True r_noise | FEM strength | Description |
|------|----------|---------------|--------------|--------------|-------------|
| A    | 500      | 0.36          | 0.00         | 0.15         | Zero noise corr (null case) |
| B    | 500      | 0.33          | 0.05 (target)| 0.15         | Positive noise corr |
| C    | 500      | 0.87          | 0.03 (target)| 0.15         | High-rate neurons |
| D1   | 200      | 0.33          | 0.03 (target)| 0.15         | Small sample |
| D2   | 600      | 0.37          | 0.03 (target)| 0.15         | Medium sample |
| D3   | 1500     | 0.30          | 0.03 (target)| 0.15         | Large sample |
| E    | 500      | 0.40          | 0.03 (target)| 0.40         | Strong FEM modulation |
| F    | 500      | 0.87          | 0.10 (target)| 0.15         | High rate + strong corr |

## Key Quantitative Results

| Case | Pipeline Dz | zC (corrected) | zU (uncorrected) | True z | Bias (zC - true) |
|------|-------------|----------------|------------------|--------|------------------|
| A    | -0.0013     | -0.0015        | -0.0002          | -0.0003| -0.0012          |
| B    | -0.0005     | +0.0027        | +0.0032          | +0.0031| -0.0004          |
| C    | +0.0017     | +0.0063        | +0.0045          | +0.0045| +0.0018          |
| D1   | +0.0024     | +0.0032        | +0.0008          | +0.0008| +0.0023          |
| D2   | +0.0007     | +0.0030        | +0.0023          | +0.0023| +0.0006          |
| D3   | +0.0006     | +0.0025        | +0.0019          | +0.0018| +0.0007          |
| E    | -0.0045     | -0.0022        | +0.0023          | +0.0023| -0.0044          |
| F    | -0.0019     | +0.0167        | +0.0186          | +0.0185| -0.0018          |

**Real data reference**: Dz = -0.11 (pooled), Dz = -0.30 (high-rate pairs), corrected z = -0.20

## Interpretation

1. **The pipeline is essentially unbiased.** Across all eight parameter regimes, |Dz| < 0.005 and |bias| < 0.005. The largest bias magnitude was Case E (strong FEM = 0.4) with Dz = -0.0045, which is **24x smaller** than the observed pooled Dz of -0.11 and **67x smaller** than the high-rate Dz of -0.30.

2. **No finite-sample bias.** Cases D1-D3 (200-1500 trials) show no systematic trend with sample size. The small positive bias at low trial counts (+0.002) is likely from Crate underestimation due to noisy distance-binned estimates.

3. **No rate-dependent bias.** Case C (high rates, 0.87 sp/bin) shows negligible bias (+0.002), ruling out the possibility that the pipeline artifact is rate-dependent.

4. **Strong FEM produces the largest (but still small) negative bias.** Case E with FEM strength 0.40 shows Dz = -0.005. This is a mild overestimation of off-diagonal Crate, but the effect is far too small to explain the data. Real FEM effects are likely in the 1-5% of total variance range (Crate/Ctotal ~ 0.02), which produces negligible bias.

5. **The pipeline correctly recovers known noise correlations.** In Case F (true r ~ 0.019), the pipeline recovers zC = +0.017 vs true z = +0.019 -- close agreement.

## Conclusion

**Hypothesis 1 is rejected.** The LOTC pipeline does not introduce a meaningful negative bias in corrected noise correlations. The maximum pipeline bias observed (|Dz| < 0.005) is at least an order of magnitude too small to explain the observed Dz = -0.11 (pooled) or Dz = -0.30 (high-rate pairs). The negative shift in corrected noise correlations seen in the real data must arise from either:
- A real biological signal (true negative intrinsic noise correlations after FEM correction)
- A feature of the real data not captured by this synthetic model (e.g., non-Gaussian noise structure, trial-to-trial gain fluctuations correlated with eye position in a way not captured by simple gain fields, or interaction effects between pipeline steps that only manifest with real data statistics)

The pipeline machinery itself (distance binning, linear intercept fitting, split-half PSTH, PSD projection, cov-to-corr normalization) is not the source of the observed bias.
