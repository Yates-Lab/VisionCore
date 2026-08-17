# Saved-cache P/Q semantic analysis

`pq_semantics.py` implements sections 4–6 of the ConvGRU registration audit.
It never imports or runs the core model. It reads only the validated saved
ConvGRU states, saved RR100 maps/readout, crossed folds, and canonical
fold-specific rank-8 bases.

For each contrast and fold, `P = U Uᵀ` is the candidate movement/output
subspace and `Q = I − P` is the complementary subspace. `P` and `Q` are
channel subspaces shared across every spatial position. They are not sets of
eight and 120 native channels.

## Stages

Run the state-semantic quantities first:

```bash
conda run --no-capture-output -n yatesfv python -m \
  paper.fig4.mechanism_audit_v1.registration_mechanism.pq_semantics \
  --stage variance --folds 0 1 2 3
```

Run the tiled-readout decomposition after those jobs finish:

```bash
conda run --no-capture-output -n yatesfv python -m \
  paper.fig4.mechanism_audit_v1.registration_mechanism.pq_semantics \
  --stage readout --folds 0 1 2 3 --device cuda:0
```

This is saved-cache computation, not a core replay. The default device is
CPU; selecting CUDA only accelerates the frozen cached tiled readout. A CUDA
run acquires the same exclusive Figure 4 GPU lock as rank-8 fitting, checks the
single authoritative ten-hour ledger between held-out pairs, and debits all
attempted wall time (including failed runs). It therefore cannot run alongside
the rank-8 continuation or escape the analysis compute cap.

Run the required descriptive probes on every fold before consolidation:

```bash
conda run --no-capture-output -n yatesfv python -m \
  paper.fig4.mechanism_audit_v1.registration_mechanism.pq_semantics \
  --stage probes --folds 0 1 2 3
```

Consolidate all completed fold/contrast products:

```bash
conda run --no-capture-output -n yatesfv python -m \
  paper.fig4.mechanism_audit_v1.registration_mechanism.pq_semantics \
  --stage consolidate --require-all-folds
```

Each expensive fold/contrast stage has a completion marker tied to the SHA-256
of its basis and is reused unless `--overwrite` is given.

## Definitions that must remain distinct

- Total motion fraction is `E||P Δh||² / E||Δh||²`. It answers where the
  complete movement-induced state change lies.
- Candidate energy per dimension is `E||P Δh||² / 8`; complementary energy
  per dimension is `E||Q Δh||² / 120`. Their ratio asks whether a candidate
  dimension is unusually movement-sensitive.
- Content variation retains channel and spatial coordinates in its grand-mean
  state. A second estimate first averages each image over trajectories and
  frames.
- Trajectory variation is across trajectories at fixed image, scale, scored
  frame, and spatial position.
- The baseline response reference is a stabilization mean formed from the
  fold’s training plus validation pairs. No crossed test pair contributes.
- `a_P = W(P h)` and `a_Q = W(Q h)` exclude the bias. The complete
  preactivation is `bias + a_P + a_Q`, with the bias added once.
- P-only and Q-only baseline reconstructions and movement interventions are
  literal states passed through the cached 1×1 feature weights, each unit’s
  14×14 tiled spatial kernel, softplus, normalized-map calculation, and SSI.
  Nonlinear rate/SSI results are never inferred from linear preactivation
  fractions.
- Complete-map recovery is primarily paired-expected-spike weighted, matching
  the validated rank-8 analysis. Equal-unit mean per-unit recovery is saved as
  an explicitly named robustness result.

## Canonical outputs

- `native_channel_leverage.csv`: every channel’s leverage, the complete
  cumulative curve, effective participation count, and associations with
  exact readout strength and held-out movement energy.
- `native_channel_leverage_overlap.csv` and
  `native_channel_leverage_topk_overlap.csv`: projector/leverage overlap and
  the full top-k curve. No leverage threshold defines a “circuit.”
- `pq_variance_decomposition.csv`: movement, content, image-mean content, and
  trajectory decompositions, with total fractions and per-dimension energy in
  separate columns.
- `pq_readout_decomposition.csv`: absolute linear variance/covariance,
  mean-rate and normalized-map consequences, training-mean baseline
  reconstructions, and full P/Q movement dose curves.
- `per_unit_pq_reliance.csv`: weight-based and activity-weighted reliance,
  baseline/movement recovery, historical SF, movement optimum, SSI benefit,
  and reversal metadata.
- `pq_supporting_arrays/`: compact bases/leverage vectors and common-scale,
  expected-spike-weighted population mean normalized maps. No individual map
  is independently rescaled.
- `pq_semantics_manifest.json`: exact input/output provenance, definitions,
  completeness, and missing fold products.

## Descriptive probes

It uses spatially pooled state features and fixed ridge regression. Continuous
shared targets—scale, path length, and signed retinal x/y displacement—use a
strict crossed outer-train to held-out-image-and-trajectory evaluation.

Image-identity labels cannot generalize to identities absent from probe
training. The image probe therefore uses only the projector’s already-held-out
test images, trains on half the held-out trajectories at 1×, and evaluates the
same identities on disjoint trajectories. It is explicitly labelled
within-heldout-image-set identification, not unseen-class decoding.

The result is saved as `pq_descriptive_probes.csv`. Decodability is
descriptive and must not be called causal.

## Saved-products-only Figure 1

After rank-8 validation and all 12 P/Q fold-by-contrast cells are finalized,
render the semantic figure with:

```bash
conda run --no-capture-output -n yatesfv python -m \
  paper.fig4.mechanism_audit_v1.registration_mechanism.plot_semantics \
  --bootstrap-draws 10000
```

`plot_semantics.py` never imports the model or opens the state, map, or readout
caches. It accepts only hash-valid finalized CSV/NPZ/JSON products, requires
all fold markers and the rank-8 held-out gate, and fails without publishing a
complete plot manifest if any source is incomplete or changes during plotting.
The output directory contains editable SVG, vector PDF, 600-dpi PNG, a caption
draft, six exact plotting tables, a labelled compact NPZ, and a SHA-256
manifest. Fold-specific estimates are used for inference; consensus
projectors remain visualization-only.
