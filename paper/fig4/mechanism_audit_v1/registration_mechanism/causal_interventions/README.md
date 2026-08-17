# ConvGRU causal transport and realignment tests

This package implements Sections 10--11 of the Overnight ConvGRU Mechanism
Audit. It does not fit a new subspace, retrain the model, or run the full
100-image by 1,000-trajectory bank.

## Frozen intervention contract

The transport stage evaluates the literal frozen ConvGRU and seven controls:

- candidate hidden-to-hidden kernel, center tap only;
- reset/update hidden-to-hidden kernels, center tap only;
- all three recurrent kernels, center tap only;
- three fixed global permutations of the eight off-center taps; and
- a no-recurrence reference that resets the hidden state at each internal
  step while retaining current-input processing.

An offset permutation moves each complete 128-by-128 channel matrix as one
object. It retains every center matrix, the multiset of off-center matrices,
and the exact recurrent-kernel norm. Checkpoint parameters are never mutated.

The realignment stage preserves the literal reset and update gates, current
candidate input, retained state, and unshifted complementary contribution. It
changes only the spatial phase of the rank-8 P-projected candidate recurrent
preactivation:

```text
a_rec' = Q a_rec + shift(P a_rec)
n'     = tanh(a_current + a_rec')
h'     = (1-z) h_previous + z n'
```

The implementation actually adds `a_rec' - a_rec` to the literal concatenated
candidate convolution. This retains the checkpoint's CUDA accumulation order
and makes the zero-shift endpoint exact. Positive `(dy,dx)` moves content down
and right. The eye-derived shift uses the empirically calibrated full 2-by-2
eye-degree to feature-pixel transform. The activation-derived cross-correlation
shift is an oracle upper bound. No consensus projector is available to the
runner: each P intervention loads the basis from the fold that held out both
the image and trajectory.

The recurrence is only the eight steps inside each independent 32-lag scored
window, ordered from newer-support to older-support. Hidden state is reset for
every one of the 40 scored outputs. The exact retinal-lag supports are:

```text
[0..17], [0..19], [0..21], [0..23],
[2..25], [4..27], [6..29], [8..31]
```

## Pilot and stopping rule

The pilot reuses the previously saved outcome-blind selection exactly:

- image positions: `1, 5, 7, 4`;
- trajectory positions: `1, 3, 4, 6, 8, 11, 12, 14, 16, 18, 20, 22`;
- transport: their complete 4-by-12 Cartesian product (48 pairs).

P-specific pilot inference uses only the 12 cells in that fixed grid that are
crossed-fold held out: fold counts `4, 6, 0, 2`. The frozen grid contains no
fold-2 image. It is left missing rather than silently selecting a post-outcome
replacement; the gated full confirmation contains 12 cells from every fold.

The 8-by-24 confirmation is authorized only when all upstream scientific gates
pass and the pilot shows all of the following:

1. exact replay identity;
2. at least one candidate/all/permuted geometry intervention attenuates all
   three historical SSI effects toward zero by at least 0.002 bits;
3. that same class of intervention retains stabilized normalized-map recovery
   of at least R2=0.25;
4. correct eye-derived and activation-oracle P shifts each improve high-SF 3x
   SSI by at least 0.002 bits;
5. opposite and random shifts do not produce a material rescue, and the
   eye-derived P rescue exceeds the Q-shift control by at least 0.002 bits; and
6. induced P misalignment impairs high-SF 1x SSI by at least 0.002 bits.

Failure stops the analysis after the pilot. It never launches a more flexible
mechanism search.

## Commands

Run from `/home/jake/repos/VisionCore`. The plan stage is CPU-only and does not
load the model:

First finish the two currently outstanding upstream products. P/Q semantics
uses only saved caches (the readout stage may use CUDA):

```bash
PYTHONPATH=$PWD conda run --no-capture-output -n yatesfv python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.pq_semantics \
--stage variance --folds 0 1 2 3

PYTHONPATH=$PWD conda run --no-capture-output -n yatesfv python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.pq_semantics \
--stage readout --folds 0 1 2 3 --device cuda:0

PYTHONPATH=$PWD conda run --no-capture-output -n yatesfv python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.pq_semantics \
--stage probes --folds 0 1 2 3

PYTHONPATH=$PWD conda run --no-capture-output -n yatesfv python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.pq_semantics \
--stage consolidate --require-all-folds
```

Then time one held-out GRU-instrumentation pair and resume all 48 fold-heldout
pairs. This stage creates both the held-out registration table and the required
synthetic shift calibration:

```bash
MPLCONFIGDIR=/tmp/fig4_mpl PYTHONPATH=$PWD \
conda run --no-capture-output -n yatesfv python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.instrument_subset \
--scope heldout --device cuda:0 --frame-batch-size 8 --max-pairs 1

MPLCONFIGDIR=/tmp/fig4_mpl PYTHONPATH=$PWD \
conda run --no-capture-output -n yatesfv python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.instrument_subset \
--scope heldout --device cuda:0 --frame-batch-size 8
```

Only after those commands finish and their gates pass, inspect the causal plan:

```bash
PYTHONPATH=$PWD conda run --no-capture-output -n yatesfv python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.causal_interventions.run_interventions \
--stage plan --scope pilot --frame-batch-size 4
```

It remains blocked until rank-8 validation, all 48 fold-held-out
instrumentation parts, and the synthetic calibration are complete and valid.
The user explicitly waived P/Q motion enrichment as an execution gate for
these required causal analyses. P/Q enrichment and the sign of the measured
alignment effect remain scientific outcomes in the final decision, but neither
can suppress the causal test once its inputs are complete.

Time one complete part in each production stage first:

```bash
PYTHONPATH=$PWD conda run --no-capture-output -n yatesfv python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.causal_interventions.run_interventions \
--stage transport --scope pilot --device cuda:0 --frame-batch-size 4 --max-pairs 1

PYTHONPATH=$PWD conda run --no-capture-output -n yatesfv python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.causal_interventions.run_interventions \
--stage realignment --scope pilot --device cuda:0 --frame-batch-size 4 --max-pairs 1
```

Then rerun those two commands without `--max-pairs`. Every completed pair has
hash-validated arrays, summary CSV, and a fail-closed completion marker, so
resumption skips only verified products. After both stages finish:

```bash
PYTHONPATH=$PWD conda run --no-capture-output -n yatesfv python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.causal_interventions.run_interventions \
--stage consolidate --scope pilot
```

Inspect `causal_interventions/pilot/pilot_to_full_gate.json`. Only when it says
`advance_to_full: true`, run the same transport, realignment, and consolidation
commands with `--scope full`.

At implementation time the shared ledger had used 4.084 of 10 GPU-hours and
had 5.916 hours remaining. This number will change; the runner always reads the
live authoritative ledger. Based on the prior exact-core throughput and the
additional recurrent replays, the conservative pre-run estimate is 0.5--2
hours for the pilot and 2--6 hours for full confirmation. The two one-part
timings are authoritative and should replace this estimate before continuing.
The runner holds the shared GPU lock, checks the deadline between frame batches
and recurrent conditions, debits failed/interrupted time, and stops at the
cumulative ten-hour boundary. Waiting to acquire the shared GPU lock is not
charged. Before a full run, every pilot marker is revalidated against the
current source/input fingerprints and the pilot decision is re-derived; a
stale `advance_to_full` file cannot authorize new work.

## Saved products

Each scope saves immutable part products plus:

```text
transport_ablation_results.csv
transport_ablation_maps.npz
realignment_results.csv
realignment_maps.npz
realignment_shift_diagnostics.csv
analysis_manifest.json
```

The map arrays contain the complete 51-by-51 mean-normalized population maps,
computed from raw expected-spike-weighted sufficient statistics; conditions
are never independently rescaled. Results CSVs contain exact SSI, mean rate,
pre-softplus/rate/normalized-map fidelity, complete normalized population-map
effect recovery for each historical contrast, all five movement scales for
transport, and the three historical SSI contrasts. Canonical copies are
published atomically under `registration_mechanism_v1/` only after every
expected part validates.

## Verification

The focused suite checks literal recurrence, center-only and no-recurrence
endpoints, norm-preserving permutation, Fourier-shift sign and norm, exact
zero-shift identity, P-only candidate modification, oracle direction, frozen
pilot identities and fold leakage, exact SSI/maps, directional pilot gates,
oracle lag-boundary rejection, current-schema instrumentation readiness, and
hash-invalidated resume markers:

```bash
python -m pytest -q tests/test_fig4_causal_interventions.py
```

No GPU/model production run was performed while implementing this package.
