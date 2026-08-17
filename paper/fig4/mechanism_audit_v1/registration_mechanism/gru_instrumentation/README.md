# Exact ConvGRU registration instrumentation

This directory implements sections 7–9 of the Overnight ConvGRU Mechanism
Audit.  It does not fit another latent space and does not rerun the 100×1,000
bank.

## Critical temporal finding

The frozen cell is one 256-input/128-hidden ConvGRU with 3×3 spatial kernels.
Its update is

```text
z = sigmoid(update_gate([x,h]))
r = sigmoid(reset_gate([x,h]))
n = tanh(out_gate([x,r*h]))
h_new = (1-z)*h + z*n
```

Thus the source's `update_gate` is the candidate-write fraction, not the
retention fraction.

For instrumentation, every recurrent preactivation is the direct bias-free
hidden-kernel operation.  Its paired current contribution is defined as the
literal fused convolution minus that recurrent term, so it includes the bias
and any small fused-versus-split floating-point accumulation correction.  The
literal fused convolution is always used for the actual gate/state update.

More importantly, the ConvGRU does **not** recur across the 40 scored movie
outputs.  Each scored output is an independent 32-lag tensor ordered from
current to oldest.  The frontend/ResNet reduce that lag axis to eight inputs,
and the cell iterates from newer-support toward older-support.  The exact
retinal-lag support unions are:

```text
[0..17], [0..19], [0..21], [0..23],
[2..25], [4..27], [6..29], [8..31]
```

Registration is therefore tested against calibrated **within-window,
backward-time** retinal displacement.  Labeling it as state carried between
the 40 outputs would be wrong.

## Storage contract

The production replay reduces all recurrent terms online:

- every step retains exact full/P/Q/readout-SVD energies as scalar arrays;
- every step retains cross-correlation/transport metrics;
- Q is evaluated as the exact native-coordinate residual `(I-UU^T)A`, without
  inventing privileged axes in the 120-dimensional complement;
- only three predeclared examples retain full maps, quantized to float16 after
  literal equation reconstruction is checked;
- no dense full-subset recurrent-term cache is written.

The required final products are `gru_projected_terms.npz`,
`registration_metrics.csv`, `gru_equation_audit.md`, and
`kernel_offset_energy.csv` under
`outputs/figures/fig4/mechanism_audit_v1/registration_mechanism_v1/`.
The fold-wise held-out benchmark is consolidated separately as
`gru_projected_terms_heldout.npz` and `registration_metrics_heldout.csv`;
those rows use fold-specific projectors and are never mixed with the complete
consensus replay that owns the canonical filenames.

## Commands

The CPU-only equation/checkpoint audit takes approximately 5–15 seconds and
does not debit GPU time:

```bash
MPLCONFIGDIR=/tmp/fig4_mpl PYTHONPATH=$PWD \
conda run --no-capture-output -n yatesfv python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.audit_checkpoint
```

It intentionally refuses the final kernel table until all 12 fold-specific
rank-8 bases exist.  `--equations-only` permits an interim equation audit and a
clearly partial kernel table.

Inspect the exact row/storage plan without loading the model:

```bash
PYTHONPATH=$PWD python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.instrument_subset \
--scope heldout --plan-only
```

Time one complete held-out pair first:

```bash
MPLCONFIGDIR=/tmp/fig4_mpl PYTHONPATH=$PWD \
conda run --no-capture-output -n yatesfv python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.instrument_subset \
--scope heldout --device cuda:0 --frame-batch-size 8 --max-pairs 1
```

Then rerun the same command without `--max-pairs`; completed pairs are skipped.
The command shares the causal-low-rank GPU lock and cumulative ledger, carries
forward all prior time, and stops between frame batches at ten cumulative
GPU-hours.  The earlier exact core replay required about 0.10 accelerator-hours
for 192 pairs; the new Fourier registration is the unknown dominant cost, so
the first-pair timing is the authoritative runtime estimate.

After fold stability authorizes consensus visualization, the complete 8×24
replay is:

```bash
MPLCONFIGDIR=/tmp/fig4_mpl PYTHONPATH=$PWD \
conda run --no-capture-output -n yatesfv python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.instrument_subset \
--scope complete-consensus --device cuda:0 --frame-batch-size 8
```

Expected exact row counts with four random controls are:

| scope | pairs | projected-term rows | registration rows |
|---|---:|---:|---:|
| fold-wise held-out | 48 | 230,400 | 1,411,200 |
| complete consensus | 192 | 921,600 | 5,644,800 |

The held-out products should occupy roughly 0.3–0.8 GB; complete-consensus
products roughly 1–3 GB, dominated by the mandated plain CSV.  The three full
consensus examples add at most a few hundred MB.  The held-out replay saves one
additional objective example: the held-out pair nearest the joint medians of
the predeclared image and trajectory features, at fixed 1× movement, frame 20,
and internal step 4.  Its fold-specific learned basis is stored inside the NPZ
so downstream plotting never reloads a fit or model.

Once every held-out product and the rank-8 gate are final, Figure 2 is built
strictly from saved products (no model, renderer, core, or cache calls):

```bash
MPLCONFIGDIR=/tmp/fig4_mpl PYTHONPATH=$PWD python -m \
paper.fig4.mechanism_audit_v1.registration_mechanism.plot_registration
```

The plotting command fails closed on partial folds, mismatched schemas, a
stopped rank-8 gate, missing exact SSI, or a missing objective example.  It
exports SVG, vector-container PDF, 600-dpi PNG, exact plotting CSV/NPZ files,
a source-hash manifest, and a caption draft under
`registration_mechanism_v1/figure2_registration/`.

## Registration estimator

Maps are spatially mean-centered per channel.  Linear multi-channel
cross-correlation is computed with zero-padded FFTs and searched over a bounded
±4-feature-pixel grid.  Frequency-domain zero padding supplies an explicit
quarter-pixel grid.  For current evidence `A` and retained evidence `B`, the
reported lag maximizes `sum A(x,y) B(x+dx,y+dy)`.  A positive sampling lag is
corrected by an equal negative content displacement, so the reported transport
is the recurrent residual lag minus the raw previous-state lag.  This matches
the calibration's current-minus-previous feature-displacement convention.

An exact-renderer synthetic translation calibration first estimates the full
2×2 mapping from eye degrees to feature pixels, including sign and axis
cross-talk.  Natural-image registration is refused if either calibration
component has R² below 0.80 or median vector error exceeds 0.5 feature pixels.

Kernel tables use PyTorch's cross-correlation convention: offset `(dy,dx)` is
the sampled hidden input `h[y+dy,x+dx]` contributing to the current output
location.  Candidate, reset, and update hidden halves are all analyzed.  The
four PP/PQ/QP/QQ energies are an exact orthogonal Frobenius decomposition;
off-center energy remains descriptive until activation-level and causal tests
agree with it.
