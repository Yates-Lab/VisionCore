# H4 Report: Global Mean Subtraction Bias in Ceye

**Date:** 2026-03-22
**Script:** `test_h4_mean_subtraction.py`
**Session:** Allen_2022-04-13 (49 neurons, 3667 windows at t_count=2)

## Hypothesis

The global mean subtraction in `Ceye = MM - E[rate] * E[rate]^T` introduces systematic bias because the time-bin weights w_t(d) in the second moment MM(d) differ from the marginal proportions p_t. The "correct" subtraction should use bin-specific weighted means rather than the global mean.

## Mechanism

`compute_conditional_second_moments` accumulates cross-trial outer products per time bin t, then pools across time bins. For distance bin d:

```
MM(d) = sum_t w_t(d) * E[S_i S_j^T | t, d]
```

where w_t(d) = n_pairs_t(d) / n_total_pairs(d).

The current implementation subtracts the global mean outer product:
```
Ceye_current(d) = MM(d) - mu_global * mu_global^T
```

The correct subtraction for isolating Crate(d) would be:
```
Ceye_correct(d) = MM(d) - sum_t w_t(d) * mu_t * mu_t^T
```

The bias at each distance bin is:
```
bias(d) = sum_t w_t(d) * mu_t * mu_t^T - mu_global * mu_global^T
```

## Results

### Part 1: Weight Variation

The weights w_t(d) do deviate from marginal proportions p_t, but modestly:

- **Max |w_t(d) - p_t|:** 0.014 (small relative to typical p_t ~ 0.009)
- **Mean total variation distance per bin:** 0.113
- **TVD range:** 0.077 to 0.150 across distance bins

The largest deviations occur at extreme distance bins (smallest and largest distances), as expected since these bins sample preferentially from trials with particular eye movement patterns.

### Part 2: Analytical Bias

The mean off-diagonal bias is approximately constant across distance bins:

- **Mean off-diagonal bias (constant offset):** +0.012 (in covariance units)
- **Std of bias across distance bins:** 0.0016
- **Range across distance bins:** 0.005

Key insight: the bias is ~87% constant across distance (0.012 mean, 0.0016 std). The constant part is absorbed by intercept fitting. Only the ~13% that varies with distance actually distorts the Ceye curve shape.

The relative deviation of the weighted mean from the global mean is small: ||mu_weighted(d) - mu_global|| / ||mu_global|| < 3.2%.

### Part 3: Corrected Pipeline

| Pipeline | Method | Dz_pooled | Dz_highrate |
|----------|--------|-----------|-------------|
| Current (global mean) | linear | -0.0904 | -0.1402 |
| Corrected (bin-specific) | linear | -0.0848 | -0.1344 |
| Current (global mean) | PAVA | -0.0925 | -0.1424 |
| Corrected (bin-specific) | PAVA | -0.0863 | -0.1350 |
| Current (global mean) | raw | -0.0906 | -0.1395 |
| Corrected (bin-specific) | raw | -0.0828 | -0.1302 |

**Correction effect (delta Dz):**
- Linear: +0.0056 pooled, +0.0058 high-rate
- PAVA: +0.0062 pooled, +0.0074 high-rate
- Raw: +0.0078 pooled, +0.0093 high-rate

### Part 4: Fraction of Observed Bias Explained

- **Pooled (Dz = -0.11):** correction accounts for 5-7% of observed bias
- **High-rate (Dz = -0.30):** correction accounts for 2-3% of observed bias

### Part 5: Edge Cases (Window Size)

| Window (bins) | Window (ms) | n_time_bins | delta_Dz | mean_TVD |
|--------------|-------------|-------------|----------|----------|
| 1 | 4.2 | 110 | +0.0022 | 0.094 |
| 2 | 8.3 | 107 | +0.0056 | 0.113 |
| 4 | 16.7 | 91 | +0.0058 | 0.126 |
| 8 | 33.3 | 70 | +0.0077 | 0.140 |
| 16 | 66.7 | 35 | -0.0171 | 0.148 |

The bias effect is small and relatively stable across window sizes (1-8 bins). At 16 bins the correction goes the wrong direction, likely due to noise from very few windows (387) and time bins (35).

## Conclusion

**Hypothesis 4 is WEAKLY SUPPORTED but INSUFFICIENT as an explanation.**

The global mean subtraction does introduce a real bias in the correct direction (the correction shifts Dz toward zero), but it accounts for only ~5% of the observed negative Dz. The corrected Dz is still -0.085 (pooled), far from zero.

The core reason is that the time-bin weights w_t(d) are very close to the marginal proportions p_t. With 107 unique time bins and relatively uniform sampling, the reweighting effect is minimal. The bias is dominated by a constant offset (the PSTH covariance itself), which is correctly handled by the intercept fitting.

The vast majority of the negative Dz (~95%) remains unexplained and must arise from a different mechanism.
