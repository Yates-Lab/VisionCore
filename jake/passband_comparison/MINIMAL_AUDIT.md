# Minimal Figure 4 claim audit

The manuscript distinguishes three questions:

1. Does passband engagement add predictive value beyond movement class and
   class-specific nonlinear path length? The completed primary comparison
   already answers this on the fixed Figure 4 image ensemble.
2. Does spectral distribution add predictive value beyond the amount of
   dynamic retinal power? Normalize engagement by total dynamic power for
   **each movie before taking the median over the 40 images**, then compare
   class + path + total power against that baseline plus normalized overlap.
3. Does the neuron's own tuning uniquely explain the result, or does this
   rule out every alternative eye-movement descriptor? Neither comparison
   establishes these broader claims.

For movie power P and the existing unit-mass, nonnegative passband W_u,
normalized overlap is q_u = sum(P W_u) / sum(P). It is invariant to a scalar
rescaling of P. It is a weighted overlap, not the fraction of power inside
a binary passband. The cached measured-minus-stabilized projections can be
used after verifying that stabilized spectral power is numerically zero.

The additional audit keeps the existing 200 histories, 725 units, 145-unit
strict subset, image reductions of the outcomes, three repetitions of
five source-trial folds, training-only spline transformations, ridge penalty
0.01, and 1,000 paired source-trial bootstrap samples within animal. The
primary structure contrast is class + path + total power versus that model
plus q. Class + path versus class + path + q is a secondary descriptor check.
No regression hyperparameters are selected from these results.

The analysis conditions on the fixed image ensemble and fitted twin and
regressions. It uses the released finite-window spectral estimator and a
simple additive regression; it does not test all possible nonlinear
structure effects or the biological mechanism. The prior, separate
held-out-canvas comparison and tuning-shuffle diagnostics remain part of
the audit record and constrain interpretation.

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 conda run --no-capture-output -n yatesfv python -m jake.passband_comparison.normalized_overlap --input-dir outputs/passband_comparison_20260914
```

This adds `normalized_overlap/` artifacts without changing the original
prediction archives. `design.json` is written before any fits; `summary.json`
contains the numerical checks, source hashes, confidence intervals, and
strata. `predictions.npz` retains held-out predictions and fold memberships.
