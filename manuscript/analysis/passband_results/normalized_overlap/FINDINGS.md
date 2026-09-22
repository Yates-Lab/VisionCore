# Minimal Figure 4 passband audit

Question: does the retinal-input descriptor add predictive information beyond movement class and distance, and does its spectral distribution contribute beyond total dynamic power?

The audit reuses the selected Figure 4 movies and responses. The only new descriptor is per-movie normalized overlap q = passband power / total dynamic power, followed by the same median over 40 images. This summarizes spectral distribution after removing scalar power changes from each movie spectrum.

| Added predictor | Baseline | Rate error reduction (95% CI) | SSI error reduction (95% CI) |
|---|---|---:|---:|
| Raw engagement (existing) | Class + flexible path length | 14.5% [5.4, 23.1] | 14.0% [8.5, 18.1] |
| Normalized overlap | Class + flexible path length | 6.0% [1.6, 10.7] | 7.5% [4.0, 11.2] |
| Raw engagement (existing) | Class + path + total power | 0.5% [-3.0, 3.3] | 2.0% [0.6, 3.2] |
| Normalized overlap: primary structure check | Class + path + total power | 0.3% [-1.8, 2.7] | 4.2% [1.6, 6.7] |

Values are median paired reductions in residual squared prediction error across the fixed 725 model units, not percentage points of total explained variance. Folds group source eye trials; each comparison uses the same held-out observations and 1,000 paired source-trial bootstrap resamples within animal. The 40-image ensemble and fitted models are fixed.

The corresponding primary structure contrast in the strict 145-unit subset is -0.2% [-2.0, 2.3] for rate and 4.3% [1.1, 7.6] for SSI.

Interpretation for the manuscript: engagement adds prediction beyond movement class and distance; the normalized spectral distribution adds modest predictive value for SSI beyond those descriptors and overall dynamic power. There is no resolved rate increment in the primary structure contrast. This is a conditional predictive result for the model and image ensemble.

The result does not identify unique neuron-to-passband routing or rule out all other eye-movement descriptors. Normalization does not undo finite-window spectral blurring, and the additive regression does not test all nonlinear structure effects. The previous tuning-shuffle result remains inconclusive about tuning specificity.

The previous separate raw-engagement test on held-out source canvases and eye trials had reductions of 11.1% [-1.4, 20.1] for rate and 0.7% [-5.4, 3.2] for SSI; both intervals include zero. The new normalized-overlap audit does not establish generalization across scenes.

Validation: 85 checks passed, including selected checkpoint/source hashes, zero stabilized power, positive denominators, invariance to movie power scaling, identical source-trial folds, and independent reproduction of every baseline fold. Maximum baseline prediction discrepancy: 2.23e-05 percentage points.

Reproduction: `jake/passband_comparison/MINIMAL_AUDIT.md`. Full new results: `summary.json`; per-unit scores: `normalized_overlap_per_unit.csv`; predictions and folds: `predictions.npz`.
