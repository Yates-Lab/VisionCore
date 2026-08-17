# Figure 4 nonlinear phase causal analysis

This directory contains the exact counterfactual experiment used to separate
the linear FEM-driven input perturbation from internal nonlinear processing in
the fitted V1 digital twin.

## Canonical run

The canonical analysis uses the already validated selection of eight images,
24 drift-only trajectories, and movement scales `0, 0.5, 1, 2, 3`. It retains
the true 31-frame history and scores 40 outputs per movie.

```bash
PYTHONPATH=. conda run -n yatesfv python \
  paper/fig4/nonlinear_phase_causal/run_experiment.py --device cuda:0

PYTHONPATH=. conda run -n yatesfv python \
  paper/fig4/nonlinear_phase_causal/analyze_and_plot.py
```

The runner saves one resumable part per image. Independent GPU workers may use
`--image-ids ... --parts-only`; a subsequent canonical command detects those
parts, verifies their signatures, merges them, and writes the run manifest.

## Conditions

- `stable`: matched stabilized movie.
- `full`: intact moving movie.
- `tangent`: exact first-order JVP of the full pre-softplus model around the
  matched stabilized movie, followed by the unchanged softplus and SSI.
- `magnitude_only`: stable SplitReLU polarity route with moving magnitude.
- `route_only`: moving polarity route with stable magnitude.
- `shuffled_route`: moving magnitude and the exact number of motion-induced
  route switches, with switch locations shuffled within each sample/channel.

The SplitReLU factorial is exact: if `a` is its signed input, `m=I[a>0]` and
`u=|a|`, then its output is `H(m,u)=[mu,(1-m)u]`. Same-source reconstruction,
zero-motion replay, JVP anchoring, and shuffled switch counts are all audited.

## Outputs

Outputs are written to `outputs/figures/fig4/nonlinear_phase_causal_v1/`:

- one paper-ready Figure 4 replacement in PDF/SVG/PNG;
- two supplemental figures in PDF/SVG/PNG;
- diagnostic figures;
- exact arrays and plotting tables;
- technical, plain-English, caption, and manuscript-text drafts;
- `run_manifest.json`, `key_statistics.json`, and `verification.json`.

The analysis deliberately treats “amplification” as an empirical signed result,
not a premise. It also limits “phase route” to positive/negative SplitReLU
branch assignment; this is not a Fourier phase-scramble experiment.
