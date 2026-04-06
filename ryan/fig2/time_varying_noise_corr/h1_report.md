# H1: Time-Resolved Covariance Analysis

Session: Allen_2022-04-13 | 49 neurons | 55 valid time bins | Trial counts: 42-78 (CV=0.190)

## Tercile Analysis

| Metric | Early (t=10..44) | Mid (t=46..80) | Late (t=82..118) | Early-to-Late Change |
|--------|:-:|:-:|:-:|:-:|
| Trial count (mean) | 76.7 | 67.4 | 49.6 | -35.3% |
| Pop mean rate | 0.4222 | 0.3717 | 0.3439 | -18.5% |
| PSTH cov (off-diag) | 0.01065 | 0.00611 | 0.00395 | -62.9% |
| Total cov (off-diag) | 0.05639 | 0.04632 | 0.02355 | -58.2% |
| Noise cov (off-diag) | 0.04574 | 0.04020 | 0.01960 | -57.1% |
| Fisher-z noise corr | 0.0952 | 0.0876 | 0.0501 | -47.4% |

## Sliding Window Spearman Correlations (window = 10 time bins)

| Quantity | rho | p-value | Significance |
|----------|:---:|:-------:|:---:|
| PSTH cov (off-diag) | -0.336 | 2.23e-02 | * |
| Noise cov (off-diag) | -0.816 | 4.76e-12 | *** |
| Fisher-z noise corr | -0.870 | 4.36e-15 | *** |
| Mean firing rate | -0.737 | 5.30e-09 | *** |

## Key Findings

1. **All covariance quantities decline monotonically with time within fixation.** PSTH covariance, noise covariance, and noise correlation all decrease from early to late time bins.

2. **Noise correlation shows the strongest and most significant trend** (Spearman rho = -0.870, p < 1e-14). It drops by ~47% from the early to the late tercile (Fisher-z: 0.095 to 0.050).

3. **PSTH covariance also declines** (-63% early-to-late), but the sliding-window trend is noisier (rho = -0.34, p = 0.02). Much of the PSTH covariance drop occurs between early and mid terciles, with a spike around time bin 50 visible in the sliding-window trace.

4. **Mean firing rate declines by ~19%** across the fixation (rho = -0.74, p < 1e-8), which likely contributes to the covariance decline via a mean-variance relationship.

5. **The noise covariance decline (-57%) exceeds what firing rate alone would predict (-19%)**, suggesting that noise correlations genuinely change with time-in-fixation, not just as a scaling artifact.

![Figure](h1_time_resolved_covariance.png)
