# Rank-8 crossed-fold validation

This directory implements only Section 3 of the Overnight ConvGRU Mechanism
Audit.  It does not repeat the rank sweep and it never reruns the image-to-core
model.

The validation is fixed to:

- rank 8, selected previously from validation data only;
- the three historical contrasts;
- folds 1--3 for new fitting, with the existing validated fold 0 reused;
- the unchanged normalized-map sufficiency/necessity objective;
- the same three predeclared initializations;
- learned-projector and rank-matched readout-SVD held-out evaluation.

## Production commands

Run from `/home/jake/repos/VisionCore`.  The dry runs are CPU-only preflight
checks and are safe to repeat:

```bash
conda run --no-capture-output -n yatesfv python -m paper.fig4.mechanism_audit_v1.registration_mechanism.rank8_validation.fit_rank8_folds --dry-run
conda run --no-capture-output -n yatesfv python -m paper.fig4.mechanism_audit_v1.registration_mechanism.rank8_validation.evaluate_rank8_folds --dry-run
```

Start the nine missing fits on GPU 0:

```bash
nohup conda run --no-capture-output -n yatesfv python -m paper.fig4.mechanism_audit_v1.registration_mechanism.rank8_validation.fit_rank8_folds --device cuda:0 > outputs/figures/fig4/mechanism_audit_v1/registration_mechanism_v1/rank8_validation/rank8_fit.log 2>&1 &
```

After fitting finishes, evaluate the learned and readout-SVD projectors and
the two directions of higher-SF cross-transfer:

```bash
nohup conda run --no-capture-output -n yatesfv python -m paper.fig4.mechanism_audit_v1.registration_mechanism.rank8_validation.evaluate_rank8_folds --device cuda:0 --frame-batch-size 40 > outputs/figures/fig4/mechanism_audit_v1/registration_mechanism_v1/rank8_validation/rank8_evaluation.log 2>&1 &
```

Finally, construct the saved tables and gated visualization-only consensus
projectors on CPU:

```bash
conda run --no-capture-output -n yatesfv python -m paper.fig4.mechanism_audit_v1.registration_mechanism.rank8_validation.finalize_rank8_validation
```

Every command is resume-safe without `--overwrite`.

## Compute boundary and expected duration

The single authoritative causal-analysis ledger carries forward every prior
GPU event and stops at 10 cumulative hours.  At preflight it contained 3.028
GPU-hours, leaving 6.972 hours before an interim report was mandatory.  Read
the current value from `causal_low_rank_v1/gpu_budget.json`; do not subtract
from the stale preflight number.  Deadline checks now also run during denominator loading,
movement-PCA construction, and CPU-resident split loading, rather than only in
the optimizer loop.

The matrix optimization itself took roughly 0.4--2.5 minutes for the saved
fold-0 rank-8 fits.  The dominant and variable cost is decompressing each
approximately 20-GiB training slice from the shared LZF HDF5 cache.  Based on
the observed fold-0 cold-load range, the nine fits should take approximately
2--6 hours; held-out evaluation and high-SF cross-transfer should add roughly
0.5--1 hour.  The upper end is close to the remaining reporting boundary, so
the scripts finish one complete fold at a time and save progress after every
fit/product.

At implementation time GPU 0 was the appropriate device (about 4.8 GiB in use
and idle); GPU 1 was heavily occupied and must not be used.  Device occupancy
must be checked again immediately before launch.

If the ledger stops the run, do not raise the limit.  Record the completed
folds with:

```bash
conda run --no-capture-output -n yatesfv python -m paper.fig4.mechanism_audit_v1.registration_mechanism.rank8_validation.finalize_rank8_validation --allow-incomplete
```

Then write the specified interim report before authorizing more accelerator
time.

## Gates

Held-out generalization is declared for a contrast only when both sufficiency
and necessity complete-map R² reach 0.40 in at least three of four folds and
both fold medians reach 0.40.

A per-contrast consensus projector is saved only when all six pairwise fold
overlaps exceed the 97.5th percentile of a deterministic rank-matched random
subspace null.  Consensus is the top eight eigendirections of the mean of the
four projectors; raw bases are never averaged.

A shared higher-SF consensus is saved only when both high-SF projectors pass
that stability gate, all four within-fold between-contrast overlaps exceed the
random bound, their median overlap is at least 0.50, and both sufficiency and
necessity cross-transfer R² reach 0.40 in both directions in every fold.

Consensus projectors are explicitly marked visualization-only.  All semantic
inference must remain fold-wise on held-out data.
