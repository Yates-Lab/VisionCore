# Passband engagement: stronger Figure 4 text comparison

This analysis reuses the selected Figure 4 spectral and response caches. It
does not evaluate or optimize the twin, or change any figure. The outcomes
are the existing measured-minus-stabilized rate and spatial SSI modulation.

The primary analysis preserves the released reduction over 40 images and
compares predictions for held-out eye histories. The secondary analysis uses
all 8,000 movies and holds out source canvases and source eye-movement trials
simultaneously. Histories from one trial and patches from one canvas never
cross their respective training/test boundary. The 725 neurons are a fixed,
pooled model population; the 145 strictly validated neurons are also reported.

## Fixed design, chosen before computing comparative results

- Five folds, with three independently seeded repetitions for the primary
  analysis. Eye-trial folds stratify animal and movement class. The secondary
  analysis crosses five canvas folds and five eye-trial folds, so every movie
  is tested once with neither its source canvas nor eye trial in training.
- Fit each neuron's outcomes separately. All models include an animal term.
  Compare movement class, path length, engagement, class plus path length,
  and class plus path length plus engagement. The class/path baseline permits
  a different nonlinear path-length relationship in each movement class.
- Continuous predictors use a training-only empirical percentile transform
  followed by a cubic B-spline with two interior knots (five nonconstant
  basis functions). Every feature is centered and standardized using training
  data only. Fit ridge regression with fixed penalty 0.01 in mean squared
  loss units; primary sensitivity analyses use 0.001 and 0.1. No hyperparameter
  or model is selected using test outcomes.
- Add total dynamic power to both nested models. A population-average
  passband and 32 random, fixed, self-excluding assignments of other neurons'
  passbands provide specificity controls. Shuffle assignments are reused
  across folds, outcomes, and scenes, and are not optimized.
- Report out-of-fold R-squared, paired changes, and reduction in prediction
  squared error. Average squared errors across repeated splits rather than
  scoring an ensemble of averaged predictions. Negative R-squared is retained.
- Primary intervals resample source eye trials within animal, conditional on
  the fixed image ensemble. Secondary intervals cross source-canvas and
  eye-trial resampling. All neurons share the same resampling weights.
  These intervals quantify sampling of stimuli/histories for this fixed twin,
  not uncertainty from neural noise, training, or a population of animals.
- Report effects in both animals, in each movement class, and in the strictly
  validated tuning subset. Report conventional partial rank correlations as
  descriptive supplements, not as substitutes for held-out prediction.

## Reproduce

Run from the repository root, using the `yatesfv` environment:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 conda run --no-capture-output -n yatesfv python -m unittest jake.passband_comparison.test_comparison -v
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 conda run --no-capture-output -n yatesfv python -m jake.passband_comparison.run --out-dir outputs/passband_comparison_20260914
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 conda run --no-capture-output -n yatesfv python -m jake.passband_comparison.estimator_diagnostics --out-dir outputs/passband_comparison_20260914
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 conda run --no-capture-output -n yatesfv python -m jake.passband_comparison.finalize --out-dir outputs/passband_comparison_20260914
```

`design.json` binds the inputs and design. Saved predictions, fold assignments,
per-unit tables, controls, and `summary.json` permit independent checks.
The manuscript addition must describe the observed result, including any
failed specificity or generalization controls.
