# Stronger passband comparison

Follow-up: the [minimal normalized-overlap audit](normalized_overlap/FINDINGS.md) separates spectral distribution from total dynamic power. It supports a modest additional SSI prediction benefit beyond movement class, path length, and total power on the fixed image ensemble.

The existing Figure 4 engagement descriptor improves held-out prediction beyond a flexible movement-class/path-length baseline when the 40-image ensemble is fixed. The result supports a text-level predictive claim. It does not establish that matching a neuron to its own tuning is uniquely explanatory. The specificity controls have limited leverage because the estimated predictors are highly similar and the spectral window is short.

## Main contrasts

All values below are the median, across the fixed 725 units, of the paired percentage reduction in held-out squared prediction error. Intervals resample source histories, with source canvases additionally resampled for the movie analysis. Positive values favor adding engagement.

| Comparison | Rate reduction (95% CI) | SSI reduction (95% CI) |
|---|---:|---:|
| Fixed image ensemble: add engagement beyond class + path | 14.5% [5.4, 23.1] | 14.0% [8.5, 18.1] |
| Fixed image ensemble: additionally control total dynamic power | 0.5% [-3.0, 3.3] | 2.0% [0.6, 3.2] |
| New canvases and eye trials: add engagement beyond class + path | 11.1% [-1.4, 20.1] | 0.7% [-5.4, 3.2] |
| New canvases and eye trials: additionally control total power | 1.0% [-1.6, 3.3] | -0.5% [-2.2, 0.6] |

## Held-out model performance

Primary R-squared uses out-of-fold predictions and the outcome variance for each neuron. The percentages above concern residual prediction error, not percentage points of total variance.

| Predictors (all also include animal) | Median rate R² | Median SSI R² |
|---|---:|---:|
| animal | -0.025 | -0.026 |
| class | 0.743 | 0.627 |
| path | 0.824 | 0.682 |
| engagement | 0.831 | 0.728 |
| class_path | 0.831 | 0.721 |
| class_path_engagement | 0.856 | 0.766 |
| class_path_dynamic | 0.855 | 0.762 |
| class_path_dynamic_engagement | 0.856 | 0.779 |
| class_path_dynamic_population_passband | 0.857 | 0.768 |
| class_path_dynamic_shuffled | 0.857 | 0.774 |

The median paired R² increments for the main contrast are 0.023 and 0.030.

## Sensitivity and specificity

- The strict 145-unit subset has error reductions of 17.3% for rate and 12.2% for SSI.
- The main increment is positive in both animals and at both additional ridge penalties. It is not uniformly positive within both movement classes: the rate gain is concentrated in drift histories.
- Engagement alone clearly improves on class alone. Its primary advantage over path length alone is supported for SSI; the rate interval spans zero.
- After controlling total dynamic power, the incremental effect is small. Own-passband matching does not improve on mean shuffled-assignment error in the primary analysis. This is an outcome of the present assay, not evidence that actual neuronal tuning is irrelevant.
- Across scenes at fixed eye history, all tested descriptors have negative median R² for the centered response variation. This limits a claim that these scalar descriptors explain scene-specific effects.

## Finite-window estimator limitation

The released 250-ms, 240-Hz estimator samples positive temporal frequencies at 4-Hz spacing and uses DPSS NW=1.5, K=2 (nominal half-bandwidth 6 Hz). In a noise-free calibration, known 1-Hz and 2-Hz temporal carriers produce normalized estimated spectra with cosine similarity 0.991; at a one-second calibration window it is 0.828. This calibration changes only the estimator window and does not replay the twin.

Actual image-reduced engagement predictors have median pairwise Spearman correlation 0.974 and median correlation 0.984 with total dynamic power. There are 252 model units (45 in the strict subset) whose preferred TF lies below the first nonzero temporal bin. These observations make the negative shuffle result inconclusive about tuning specificity. They do not establish that estimator resolution caused the result; shared stimuli and broad or similar tuning can also produce correlated predictors.

## Methods correction

Engagement uses nonnegative fitted Yu SF×TF predictions multiplied by measured axial-orientation weights, normalized to unit mass; it does not use raw phase-RMS responses when passband_weight is present.

## Reproduction and review

- Code and fixed statistical design: `jake/passband_comparison/README.md`.
- `summary.json`: complete unrounded results, strata, regularization controls, and estimator diagnostics.
- `primary_per_unit.csv` and `secondary_per_unit.csv`: per-neuron scores and paired contrasts.
- `per_source_canvas.csv`: results for each held-out source canvas.
- Saved prediction archives contain fold memberships and all held-out predictions; shuffle squared errors and assignments are retained.
- Five scientific tests passed. Final numerical audit: 104 passed checks; fresh regression fits agree within 7.6e-06 percentage points.

## Scope of inference

- Model responses, not measured neural-noise discrimination or attention.
- Primary estimates condition on the fixed 40-image ensemble.
- Bootstrap intervals condition on fitted cross-validation models and resample source histories/canvases, not model fits or animals.
- Population-average engagement control averages image-reduced unit predictors in the primary analysis; it is a common spectral predictor, not a raw uniform-power measure.
- A negative tuning-shuffle comparison is conditional on a finite-window spectral estimator; it cannot rule out actual tuning-specific effects. Estimator resolution and predictor collinearity were inspected but not established as the cause of the negative result.
- Specificity controls establish predictive value relative to tested descriptors, not a unique causal mechanism.
