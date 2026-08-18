# M77: selected native-240-Hz mechanistic twin

## Frozen artifacts

- Checkpoint: `/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/dekel240/D240M77c_dekel_native240_freqmasked_floor0p5_gnlrnalpha0p1_presplit_s201/analysis_candidates/epoch=279-val_bps_overall=0.5912.ckpt`
- Checkpoint SHA-256: `36e51bf51593286a4dc3a330bdf6841f252fad3ed62b855ffb4932a7fe2cb853`
- Model config: `experiments/model_configs/dekel_capacitymatched_freqmasked_readoutfloor0p5_groupnorm_lrnalpha0p1_mlp_behavior.yaml`
- Model-config SHA-256: `acaead3cbfa5656e8964c26fc72feaebb86bc7c5c58d779f25e83472ffe4d5d8`
- Dataset config: `paper/model_selection/configs/multi_240_long_split3_dekel35.yaml`
- Dataset-config SHA-256: `05ccdcffb8d94d0a0c627d167394aeb9d65b9754da99fb3eddc7a9629fb6c8e1`
- Fixed Figure-4 population: `outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints`

M77 is the selected member of the native-240-Hz normalization sweep. It was
chosen for external FixRSVP generalization, not for the highest noisy rotating
validation checkpoint. The model has a causal 60-frame/250-ms visual history,
a 35 by 35 retinal crop, no recurrence, a Dekel-style feedforward core, a
42-dimensional behavior MLP/FiLM branch, and a Gaussian readout. GroupNorm is
followed by sign-preserving local response normalization with `alpha=0.1`.
Temporal and first-layer spatial filters are Hann-masked in the frequency
domain with twofold FFT padding. The readout-width proximal floor is 0.5.

The historical RR100 population contains channels defined from the Twin's
756-channel canonical union. M77 exposes 725 of those channels. Three missing
representatives are replaced by available members of the same redundancy
clusters using the frozen population specification; two clusters have no M77
member and remain explicitly inactive. Thus Figure 4 uses 98 active units and
never silently changes the population denominator.

## Prediction comparison

All Twin comparisons use identical observations and data-defined support. The
FixRSVP reliability ceiling is computed once per neuron from the data and then
shared by both models. The independent audit reconstructs support, CCabs,
500-split CCmax, the stability mask, and `CCnorm = CCabs / CCmax`; the maximum
identity error is zero across all 24 sessions and 1,703 units.

| Gate | Twin | M77 | M77 minus Twin |
|---|---:|---:|---:|
| Common-support validation BPS, overall | 0.54616 | 0.49541 | -0.05075 |
| Common-support validation BPS, unit median | 0.49885 | 0.44038 | -0.04976 paired median, 95% CI [-0.05725, -0.04281] |
| FixRSVP CCabs, stable-unit median | 0.44730 | 0.44277 | -0.01227 paired median |
| FixRSVP CCnorm, stable-unit median | 0.64023 | 0.62868 | -0.01830 paired median, 95% CI [-0.04587, 0.01875] |
| FixRSVP affine single-trial R2, unit median | 0.02236 | 0.02245 | -0.00023 paired median, 95% CI [-0.00137, 0.00131] |
| FixRSVP raw Poisson BPS, aggregate | 0.10817 | 0.10781 | -0.00036 |
| FixRSVP affine Poisson BPS, aggregate | 0.15868 | 0.16991 | +0.01123 |

The aggregate affine BPS gain is not a population-level win: the paired
per-unit median difference is -0.00184 with 95% CI [-0.00756, 0.00501]. The
correct conclusion is FixRSVP likelihood parity, near-parity in CCnorm and
single-trial R2, and a real in-distribution validation-BPS deficit.

M77 predicts every native 4.17-ms bin. Figure-3 comparisons sum adjacent
prediction and observation bins `[0,1], [2,3], ...` to the canonical 120-Hz
grid. Every session is rejected unless the rebinned observations are exactly
equal to the cached Figure-3 observations. The independent 24-session audit
passes this equality gate. Native-240 medians are rho 0.35729, raw BPS 0.06790,
and positive-affine BPS 0.14615.

## Interpretability gates

M77's population Jacobians are localized and smooth: median temporal
second-difference ratio 0.3265, median temporal power at or above 60 Hz 0.01185,
and median spatial high-frequency fraction 0.0353. On matched physical
contexts, the Twin's exemplar Jacobians are visibly checkerboarded and roughly an
order of magnitude rougher.

The periodic tuning probe uses a cycle-valid half-octave spatial grid from
1.07 to 15.94 cycles/degree and temporal frequencies from 1 to 90.5 Hz. It
stores the full measured response surface, response-weighted centers, and
explicit boundary-censoring flags; it does not interpret a boundary maximum as
a resolved preference. The 98 active RR100 units form response-weighted spatial
frequency tertiles of 32 lower, 34 middle, and 32 upper units; two inactive
channels remain explicit.

On the exact 12-unit panel frozen by the Twin response-subspace analysis and
with 128 gradient contexts for each model, M77's rank-8 model captures 0.4257
of median Jacobian energy and explains median held-out response R2 0.3366
(minimum 0.1780). Rank 16 captures 0.5698 of median Jacobian energy and
explains median response R2 0.3138. The paired basis atlas shows markedly
smoother and more localized M77 filters, but M77 does not have uniformly lower
gradient-covariance rank: the distinction is clean basis geometry rather than
a claim of globally lower intrinsic dimensionality. No tested rank reaches the
deliberately stringent 0.8 surrogate gate.

## Production Figure 3

The schema-v7 cache passes exact neuron-order, data-support, CCmax, CCnorm,
observed-FEM-term, and shared-anchor invariants. In the canonical 19-session
population, median CCnorm is 0.656 full, 0.655 with the separate extraretinal
branch zeroed, and 0.599 with the retinal movie stabilized. Median captured
rate variance is 0.247 full and 0.059 stabilized. The empirical, full-model,
and stabilized FEM-fraction medians are 0.736, 0.675, and 0.210. The full model
passes the predeclared paired equivalence test to the empirical distribution at
margin +/-0.1; the stabilized model does not.

The production bundle is in
`outputs/dekel240_paper/m77_epoch279/production_figure3/final/`.

## Figure-4 mechanism and claim boundary

The production replay expands retained 120-Hz eye traces to the native 240-Hz
model grid before movie rendering, holds gaze through the 59-frame causal
prefix, and integrates expected spikes with `dt=1/240`. Across 100 images by
1,000 traces, measured retinal motion increases expected-spike-weighted pooled
SSI by 6.86% (paired image-bootstrap 95% CI [5.84%, 7.88%]); 96/100 image
effects and all 1,000 trace-pooled effects are positive. Exact layer replay on
the separate 8-image by 8-trace mechanistic subset localizes the sequence:
motion increases temporal-filter drive; the temporal stem alone changes
spatial information by -0.39%; spatial stages 1/2/3 change it by +0.22%,
+4.54%, and +7.17%; and the RR100 output changes by +2.88%. About 59.3% of
exact temporal-stem drive lies in channels with measured median TF at least
10 Hz. The layer values are localization evidence, not production-scale effect
estimates.

The original framewise `TF = |k dot v(t)|` histogram is withdrawn as a retinal
power estimate. It discards temporal ordering and displacement correlations and
is not the temporal PSD of a translated image under drift or Brownian-like eye
motion. The corrected analysis uses the complete trajectory carrier for every
image Fourier mode,
`R_k(t) = I_k exp(-i 2 pi k.X(t))`, removes its finite-trace static component,
and estimates the folded two-sided temporal spectrum with two DPSS tapers. The
333-ms interval has 3-Hz native frequency resolution; frequencies below that
are displayed as unresolved rather than inferred from an instantaneous
velocity.

The ideal image-FFT times trajectory-phase calculation was checked against
FFTs of movies rendered through the actual finite-crop, bilinear retinal
sampler. In the 4-image by 8-trace validation, aggregate spectral-shape cosine
similarity was 0.988 and total-variation distance was 0.073; pairwise medians
were 0.891 and 0.191. Thus the trajectory-phase approximation is accurate for
the aggregate second-order spectrum, while direct rendered replay remains the
causal response test.

The primary production spectral analysis measures the spatial and temporal FFT
of movies rendered through the exact 151-pixel crop and bilinear sampler; the
ideal trajectory-phase calculation is a cross-check, not a substitute. On 100
images by 16 evenly sampled traces, total dynamic power has median within-unit,
across-image Spearman rho 0.366 with the causal moving-minus-stabilized rate
change (unit-bootstrap 95% CI [0.319, 0.396]). Raw joint
SF-by-TF-by-orientation passband-weighted power is nearly as predictive at
0.358 ([0.310, 0.374]). Dividing out total dynamic power reverses the result:
the normalized joint fraction is -0.178 ([-0.258, -0.105]) and the normalized
TF-marginal fraction is -0.147 ([-0.186, -0.112]). Extra image selectivity from
joint versus separable alignment only trends with SSI gain across all units
(rho 0.188, p=0.064, bootstrap interval [-0.004, 0.363]) and disappears in the
22 trusted-peak units (rho -0.016, p=0.942). The supported mechanism is thus
two-stage: retinal motion supplies dynamic power that enters the measured
passband and drives activity, while later spatial nonlinearities reshape that
drive into SSI. The data do not support the stronger claim that the *fraction*
of power assigned to fine joint tuning predicts which units sharpen most.

Peak TF is now defined by a local joint quadratic in log2(SF) and log2(TF)
around the measured two-dimensional maximum, including the SF-by-TF
interaction. The earlier global Gaussian center is retained only as a visible
bias diagnostic: a high global held-out R2 can be obtained by fitting the many
low-response bins while displacing the actual peak. The corrected audit has 22
trusted, 6 usable-with-caution, 2 metric-dependent, and 68 unstable or censored
units. The trusted estimates range from 2.04 to 39.59 Hz. They still separate
between 3.83 and 7.50 Hz, but the probe grid is only half-octave spaced, so this
gap is reported descriptively rather than interpreted as a discrete biological
class boundary.

The complete causal bank and stabilized control are in
`outputs/dekel240_paper/m77_epoch279/figure4_real_trace_production_corrected/`.
The combined production mechanism figure is under
`m77_two_stage_mechanism_direct/`; renderer-faithful and trajectory-phase
population audits are under `direct_rendered_joint_engagement/` and
`image_specific_joint_engagement_phase_spectrum/`; the separate harmonic and
stage-localization figure is under `nonlinearity_tuning_ssi/`.

## Verification

- Clean review-branch suite excluding the pre-existing script-style
  `tests/test_frozencore_pipeline.py`: 288 passed, 4 skipped. Three skips are
  sweep-only config comparisons whose inputs intentionally remain on the full
  archive branch; the fourth is pre-existing.
- Every changed Python source compiles successfully.
- `git diff --check` is clean.

The excluded file is an old parameterized script collected as pytest tests with
nonexistent fixtures. It is unrelated to M77 and is not modified as part of
this selection.
