# Figure 4 causal low-rank ConvGRU analysis

This directory implements the preregistered causal channel-subspace test in four isolated stages.

1. `prepare_analysis.py` freezes folds, contrasts, ranks, seeds, tolerances, and the compute budget without running the model.
2. `cache_exact_states.py` performs the single authorized corrected-history core replay and creates resumable exact-state/map caches.
3. `validate_interventions.py` is a cache-only rank-zero/full-rank integrity gate; optimization refuses to start without a full pass.
4. `optimize_subspaces.py` fits shared QR-parameterized channel projectors to complete normalized RR100 maps using training/validation identities only.
5. `evaluate_subspaces.py` computes held-out causal metrics, baselines, rank selection, uncertainty, cross-scale/contrast tests, and saved figure maps.
6. `plot_results.py` and `write_report.py` consume saved products only.

The frozen readout path is always the exact 1×1 channel mixing followed by the unit-specific 14×14 valid spatial convolution, bias, and softplus. SSI is never an optimization target.
