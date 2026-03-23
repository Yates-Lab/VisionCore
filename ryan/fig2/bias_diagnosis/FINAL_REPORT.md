# Figure 2 Noise Correlation Bias: Diagnosis and Fix

**Date:** 2026-03-22
**Session tested:** Allen_2022-04-13 (49 neurons, 3667 windows)

---

## Background: How the Shuffle Null Works

The noise correlation analysis compares two estimates of noise covariance:

- **CnoiseU** (uncorrected): `Ctotal - Cpsth`, where `Cpsth` is the
  stimulus-driven (PSTH) covariance estimated by split-half cross-validation.
- **CnoiseC** (corrected): `Ctotal - Crate`, where `Crate` is the
  eye-conditioned rate covariance estimated by trajectory-matching.

The difference `Dz = fz(CnoiseC) - fz(CnoiseU)` measures how much FEM
correction changes noise correlations, reported in Fisher-z space.

To validate the method, we run a **shuffle control**: eye trajectories are
randomly permuted across trials, destroying any spike-eye coupling while
preserving the marginal spike statistics. Under this null, the trajectory-
matching pipeline has no real eye-spike coupling to exploit. It still
conditions on eye distance and estimates a second moment matrix — but because
the eye labels are now random with respect to spikes, the resulting `Crate_shuff`
should converge to the PSTH covariance (the only systematic structure
remaining in the spike counts). Since `Cpsth` also estimates the PSTH
covariance, we expect:

```
Crate_shuff ≈ Cpsth   →   CnoiseC_shuff ≈ CnoiseU   →   Dz_shuff ≈ 0
```

**The problem:** The observed shuffle null showed `Dz_shuff = -0.007` instead
of 0 (p < 0.0001 over 50 iterations). This means `Crate_shuff` systematically
exceeded `Cpsth`, even though both should be estimating the same quantity.

---

## The Two Estimators Side by Side

Both `Crate_shuff` and `Cpsth` target the PSTH covariance, but they compute
it differently:

### Estimator 1: Trajectory-matching (`Crate`)

`estimate_rate_covariance` computes:

```
Ceye(d) = MM(d) - Erate × Erate^T
```

where `MM(d)` is the second moment matrix from `compute_conditional_second_moments`.
This function iterates over time bins t, selects all trials in that bin, and
forms every distinct cross-trial pair (i, j) whose eye-distance falls in bin d.
A time bin with n_t trials contributes up to n_t(n_t−1)/2 pairs. The second
moment is then the average over all such pairs across all time bins:

```
MM(d) = (1 / total_pairs) × Σ_t Σ_{i<j ∈ t, dist ∈ d} S_i S_j^T
```

Under the shuffle null (no real eye-spike coupling), the distance conditioning
becomes irrelevant — the curve is flat — and MM converges to the PSTH second
moment, weighted by how many pairs each time bin contributes.

### Estimator 2: Split-half (`Cpsth`)

`bagged_split_half_psth_covariance` splits trials within each time bin into
two halves (A and B), computes per-bin means, and forms the cross-covariance:

```
Cpsth = (1 / T) × Σ_t (μ_A,t - μ̄_A)(μ_B,t - μ̄_B)^T
```

Each time bin contributes exactly one row to the cross-product, regardless of
how many trials it contains. This was **uniform (1/T) weighting** over time
bins.

### The critical difference

Under the shuffle null, both estimators target the same quantity (PSTH
covariance), but with different weighting over time bins:

| Estimator | Weights time bins by | Character |
|-----------|---------------------|-----------|
| `MM(d)` (trajectory-matching) | n_t(n_t−1)/2 pairs | ~n_t² (quadratic) |
| `Cpsth` (split-half, old) | 1 per bin | uniform (1/T) |

These two weightings produce the same answer only when n_t is constant across
time bins. When it is not — and in the fixRSVP paradigm, it is not — the
estimators diverge.

---

## Root Cause: Weighting Mismatches

The bias had two components, both stemming from the same underlying issue:
the trajectory-matching estimator implicitly uses pair-count weighting (~n_t²),
but the old code used different weighting in two other places.

### Mismatch 1: Erate in the Ceye computation

`Ceye = MM - Erate × Erate^T` should use the same weighting for both terms.
But the old code computed `Erate` as:

```python
Erate = torch.nanmean(SpikeCounts, 0)   # trial-count-weighted (~n_t)
```

This is a simple average over all windows, weighting each time bin by n_t
(one entry per trial). Meanwhile, `MM` weights time bins by n_t(n_t−1)/2
(~n_t²). The result is that `Ceye` mixes **pair-weighted second moments**
with a **trial-weighted mean squared** — not a proper covariance under any
single weighting scheme.

Algebraically, the code computes:

```
Ceye_mixed = E_pair[S S^T] - μ_trial μ_trial^T
```

instead of the consistent pair-weighted covariance:

```
Ceye_proper = E_pair[S S^T] - μ_pair μ_pair^T
```

The error is:

```
Ceye_mixed - Ceye_proper = μ_pair μ_pair^T - μ_trial μ_trial^T
```

This difference is **positive semidefinite** (it is an outer-product
difference). Pair-weighting emphasizes early time bins, which tend to have
higher rates (onset transients), so `μ_pair > μ_trial` for most neurons.
The off-diagonal entries of `Ceye` are therefore systematically inflated.

### Mismatch 2: Cpsth weighting

Even with a consistent `Erate`, the split-half `Cpsth` estimator used uniform
(1/T) weighting over time bins, while `Crate` is pair-count-weighted. Under
the shuffle null, both estimate the PSTH covariance — but under different
weighting schemes. For `Dz_shuff` to be exactly zero, they must use the same
weights.

---

## How the Bias Propagates to Dz

### Why n_t varies across time bins

In the fixRSVP paradigm, fixation durations vary across trials. When
trial-aligned data is organized into a (n_trials × n_time) matrix, early time
bins contain all trials (every trial is fixating at onset), while late bins
contain only trials with long fixations. The result is a monotonically
decreasing staircase: n_t ranges from 42 to 78 (CV = 0.19) after filtering.

### Propagation chain

**(1) Inflated Ceye passes through intercept fitting unchanged.** The
inflation is present at every distance bin (it is a weighting issue, not a
distance-dependent effect), so the linear intercept captures it in full.

**(2) Under the shuffle null, the inflated Crate_shuff exceeds the
uniform-weighted Cpsth:**

```
Crate_shuff (pair-weighted PSTH cov) > Cpsth (uniform-weighted PSTH cov)
```

**(3) CnoiseC_shuff = Ctotal - Crate_shuff is then systematically smaller
than CnoiseU = Ctotal - Cpsth.** Subtracting a larger rate covariance from
the same total covariance yields a smaller noise covariance.

**(4) Cov-to-corr normalization amplifies the bias ~2.7×.** The covariance-
space difference of ~0.003 becomes ~0.007 in Fisher-z space because the
variance normalization is nonlinear.

**(5) Result:** `Dz_shuff ≈ -0.007` instead of 0.

### Quantitative verification

Under the shuffle null, we computed PSTH covariance analytically under each
weighting scheme and compared to the empirical `Crate_shuff`:

| Weighting | Off-diag mean | Description |
|-----------|---------------|-------------|
| Uniform (1/T) | 0.0079 | What old split-half `Cpsth` computed |
| Pair-count consistent | 0.0085 | Proper pair-weighted covariance |
| Mixed (pair outer, trial mean) | 0.0111 | What the old `Ceye` code computed |
| Empirical Crate_shuff | 0.0121 | Observed under shuffle |

The "mixed" analytical prediction (0.0111) closely matches the empirical
`Crate_shuff` (0.0121), confirming the weighting mismatch as the dominant
mechanism. The residual (0.001) is attributable to the intercept fitting step.

![Weighting mechanism diagnostic figure](weighting_mechanism_figure.png)

**Panel A:** The second moment `MM` weights each trial pair by n_t−1 partners
available in that time bin, creating quadratic (~n_t²) emphasis on early bins
with many trials. **Panel B:** The old `Erate` weighted each trial equally
(uniform), giving linear (~n_t) emphasis. **Panel C:** Per-neuron difference
between pair-weighted and trial-weighted mean rates. **Panel D:** Shuffle null
Dz distributions before and after the Erate fix (prior to the Cpsth fix).

---

## The Fix

Two changes bring all estimators into consistent pair-count weighting:

### Fix 1: Pair-count-weighted Erate (`estimate_rate_covariance`)

Replace the trial-count-weighted mean with a pair-count-weighted mean, so that
`Erate` uses the same weighting as `MM`:

```python
# Old (trial-count-weighted):
Erate = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy()

# New (pair-count-weighted, matching second moment):
unique_times = np.unique(T_idx.detach().cpu().numpy())
weighted_sum = torch.zeros(C, device=SpikeCounts.device, dtype=torch.float64)
total_pairs = 0.0
for t in unique_times:
    mask = (T_idx == t)
    n_t = mask.sum().item()
    if n_t < 10:
        continue
    n_pairs_t = n_t * (n_t - 1) / 2
    mu_t = SpikeCounts[mask].mean(0).to(torch.float64)
    weighted_sum += n_pairs_t * mu_t
    total_pairs += n_pairs_t
Erate = (weighted_sum / total_pairs).detach().cpu().numpy()
```

### Fix 2: Pair-count-weighted Cpsth (`bagged_split_half_psth_covariance`)

Add a `weighting` parameter to the split-half estimator. When set to
`'pair_count'` (now the default), each time bin's cross-product is weighted by
n_t(n_t−1)/2 instead of 1/T:

```python
# Old (uniform):
C_k = (XA_c.T @ XB_c) / (n_time - 1)

# New (pair-count-weighted):
pair_counts = np.array([n_t * (n_t - 1) / 2 for each time bin])
w = pair_counts / pair_counts.sum()
C_k = (XA_c * w[:, None]).T @ XB_c
```

This ensures that `Cpsth` and `Crate` weight time bins identically, so they
estimate the same weighted PSTH covariance and cancel exactly under the
shuffle null.

---

## Fix Validation

|                     | Original  | Fixed     | Change   |
|---------------------|-----------|-----------|----------|
| Real data Dz        | -0.0855   | -0.0819   | +0.004   |
| Shuffle null Dz     | -0.0068   | +0.0010   | +0.008   |
| Shuffle null p-value | < 0.0001  | 0.443     |          |
| |Signal / Null|     | 12.5x     | 83.7x     |          |

The fix eliminates the shuffle null bias: `Dz_shuff` shifts from -0.0068
(p < 0.0001) to +0.0010 (p = 0.443, not significantly different from zero).
The real-data signal changes by only +0.004 (from -0.0855 to -0.0819),
preserving the scientific conclusion. The signal-to-null ratio improves from
12.5x to 83.7x.

---

## Supporting Evidence: Hypothesis Tests

Before identifying the weighting mismatch, six hypotheses were tested
systematically to rule out other potential sources of bias.

### H1: Pipeline introduces bias on synthetic data — REJECTED

Generated synthetic Poisson data with known PSTH, FEM gains, and intrinsic
noise correlations. Pipeline bias was |Dz| < 0.005 across 8 parameter
regimes — at least 20x smaller than the observed effect.

### H2: Intercept fitting overestimates off-diagonal Crate — REJECTED

Four intercept methods produce Dz within 0.0012 of each other. Even with
no fitting at all (raw first bin), Dz = -0.091. The bias is in the Ceye
curves at every distance bin, not introduced by fitting.

### H3: PSD projection asymmetry — REJECTED (attenuates)

PSD projection actually *reduces* the apparent Dz from -0.155 to -0.086 by
clamping one large negative eigenvalue (-1.54) in CnoiseC. PSD is
conservative.

### H4: Global mean subtraction bias — WEAK (~5%)

Accounts for only ~5% of the bias (+0.006 shift). This hypothesis pointed
in the right direction — it was detecting a piece of the weighting mismatch
— but captured only a small part of the mechanism.

### H5: Jensen's inequality in normalization — REJECTED

Diagonal CV is only ~4.5%, producing a negligible Jensen contribution
(+0.005, ~6% of effect).

### H6: Shuffle null biased / Crate > Ctotal spurious — REJECTED

Under both global and within-time-bin shuffles, Crate dropped from 0.064
to ~0.010, never exceeding Ctotal in 100 iterations. The Crate > Ctotal
excess requires real spike-eye coupling. The shuffle null Dz of -0.006
confirmed the residual method bias that led to identifying the weighting
mismatch.

---

## Files

| File | Description |
|------|-------------|
| `test_h1_synthetic_ground_truth.py` | H1: Synthetic pipeline bias test |
| `test_h2_intercept_bias.py` | H2: Intercept fitting Monte Carlo |
| `test_h3_h5_psd_normalization.py` | H3+H5: PSD projection and Jensen's |
| `test_h4_mean_subtraction.py` | H4: Global mean subtraction bias |
| `test_h6_shuffle_and_crate_excess.py` | H6: Shuffle null and Crate excess |
| `test_shuffle_null_shift.py` | Focused diagnosis of shuffle null shift |
| `test_weighting_mismatch.py` | Verification of weighting mechanism |
| `test_weighting_fix.py` | Validation of the fix |
| `h1_report.md` through `h6_report.md` | Individual hypothesis reports |
| `shuffle_null_shift_report.md` | Shuffle null shift diagnosis |
| `weighting_mismatch_report.md` | Weighting mechanism verification |
| `weighting_fix_report.md` | Fix validation results |
| `SYNTHESIS.md` | Intermediate synthesis (pre-fix) |
| `make_weighting_mechanism_figure.py` | Script for weighting mechanism diagnostic figure |
| `weighting_mechanism_figure.pdf` | 2x2 diagnostic figure (weighting, rates, shuffle null) |
| `FINAL_REPORT.md` | This document |

Fixes applied to `VisionCore/covariance.py`:
- `estimate_rate_covariance` (lines 607-640): pair-count-weighted Erate
- `bagged_split_half_psth_covariance` (lines 485-590): `weighting` parameter, default `'pair_count'`
