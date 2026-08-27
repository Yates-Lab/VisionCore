# Figure 4 production contract

This is the authoritative rebuild path for the current eight-panel Figure 4.
Every Python command must run in the `yatesfv` environment with
`--no-capture-output`. The released model, checkpoint digest, exact unit
identities, eye-trace filter, and input tables must be explicit artifacts; no
script may infer or substitute them from a model nickname.

The production entry point declares every executable root in
`run_production_figure4.py`; `paper/production_source_closure.py` follows their
transitive repository-local imports and the manifest hashes that exact graph.
No directory wildcard defines the release boundary. Run
`paper/audit_production_source_closure.py` before handoff; it fails if a Python
module in this Figure 4 directory is outside the declared graph.

Shared measurement code is intentionally narrow: `eye_trace_filter.py`
implements the audited DDPI filter, `retinal_replay.py` renders retinal movies
and causal histories, `spectral_power.py` computes rendered SFxTF power and
passband projections, and `population_response.py` computes the Panel B
population estimands and crossed bootstrap. Production builders import these
primitives rather than carrying private variants.

## Scientific narrative

1. **A:** A filtered real fixation and its lag-aligned stabilized
   counterfactual are replayed through the same model. One disclosed
   illustrative unit shows increased rate and SSI.
2. **B:** All 725 checkpoint-available exact readouts summarize rate and SSI
   modulation versus filtered 250-ms path length. There is no SF or
   microsaccade split.
3. **C:** Equal-dynamic-mass conditional Kuang spectra show drift-rich and
   rapid-transient fixation distributions separately; the only in-panel text
   marks the shift toward higher temporal frequency.
4. **D:** Two raw, low-resolution, preferred-direction F0 SF×TF surfaces show
   the measured model responses. Their half-max lassos come from Yu R0/R1 fits
   to those exact surfaces.
5. **E:** Transparent filled passbands from all 725 checkpoint-available exact
   readouts show population occupancy in SF×TF space. The 145-unit strict
   validation subset supplies the audited exemplars in D, but is not silently
   substituted for the declared all-unit population.
6. **F:** The explicitly signed rapid-minus-drift power contrast is shown with
   the same two authoritative example-unit lassos.
7. **G:** Across the same 725 checkpoint-available exact readouts, passband engagement is related
   directly to measured-minus-stabilized rate and SSI modulation. Both panels
   share the same percentage y-axis. The text, not the figure, reports the
   matched comparison with path length.
8. **H:** Top-passband movies are traced through cumulative trained readout
   branches (S1 plus phase, then S2, then the ordinary output). Both displayed
   quantities are gain-invariant. No affine, tangent, ablated, or synthetic
   reference network is permitted.

## Non-negotiable measurement contracts

- Eye traces are filtered on the uniformly sampled raw-DDPI grid with the
  audited zero-phase 20-Hz-passband/30-Hz-stopband IIR, then sampled at the
  model's native 240 Hz.
- Stabilization is anchored to the frame at the model's resolved mean
  first-layer peak lag. The measured and stabilized histories therefore share
  the most influential current retinal frame.
- The tuning assay is a controlled 240-Hz drifting-grating replay through each
  checkpoint-native `(session, cid)` readout. Recorded gratings constrain
  biological SF only; they are not presented as a recorded-neuron TF assay.
- The strict tuning-validation subset requires exact identity and a complete grid, phase
  convergence, positive blank-subtracted F0 drive, an interior coherent raw
  peak, full-grid Yu R0/R1 fit quality and raw-peak agreement, repeatability at
  a second contrast, reliable recorded-neuron SF, and recorded/twin SF
  agreement.
- Every exemplar tuning surface and lasso must also pass the page-by-page
  visual audit. The all-unit density and population analyses retain all 725
  checkpoint-available exact readouts with finite Yu fits and disclose the 580
  strict-validation failures rather than relabeling them as validated. A
  displayed heatmap, lasso, centroid, and label must retain the same
  `source_unit_index`.
- Each conditional Kuang spectrum is normalized to equal TF>0 mass before
  averaging. The complete TF=0 plus TF>0 carrier budget must independently sum
  to one, so the visualization cannot mistake redistributed power for created
  power.
- Panel G uses the directly rendered movie spectrum and the exact response
  matrix for the same image/trace pair. Cached replay is allowed only when its
  SF×TF passband tensors are numerically identical to the released contract.
- Panel H must reconstruct the ordinary output numerically from cumulative
  trained readout branches. It reports temporal modulation normalized by mean
  response and spatial information in bits per spike; raw mean gain is not the
  stagewise estimand.

## Release sequence

1. Run and audit the exact-CID drifting-grating measurement at the primary and
   repeat contrasts with `run_exact_cid_drifting_tuning.py` and
   `audit_exact_cid_drifting_tuning.py`.
2. Build the one-hot exact-unit validation contract with
   `build_exact_cid_figure4_contract.py`, then the explicitly labeled all-unit
   response contract with `build_all_available_population_spec.py` and the
   matching all-unit Yu passband view with
   `build_all_unit_yu_tuning_view.py`. The latter retains and discloses strict
   validation failures rather than selecting on them. RR clustering or
   pooling is forbidden.
3. Build the filtered fixation bank, the all-unit 40-image × 200-trace response
   matrix, Panel A exemplar audit, and Panel B reduction.
4. Build the equal-mass Kuang/Rucci ensemble and the two image-sharded exact
   spectral replays. Use `compare_passband_path_length.py` for the matched
   within-unit text statistic.
5. Run `analyze_top_passband_stage_trajectory.py` on at least 10 images × 10
   traces from the released population. This minimum is 100 crossed movies;
   smaller runs are smoke tests only.
6. Run `run_production_figure4.py` to verify the declared inputs and compose
   the locked 12 × 10 inch A--H layout. A release-ready run writes canonical
   `figure4.pdf`, `figure4.png`, and `figure4.svg` outputs.
7. The runner invokes `audit_revised_figure4_release.py`, which fails closed on checkpoint,
   layout, filtering, lag alignment, population, source identity, visual tuning
   audit, power conservation, cached passband equivalence, direct response
   definitions, actual-network-only Panel H, and the 100-movie minimum.
8. Only after that audit passes does the runner invoke
   `build_figure4_results_provenance.py` to bind every numerical statement in
   `figure4_results_and_caption.tex` to hashed production artifacts.

## Claim boundaries

- Panel A is illustrative and cannot support population inference.
- Panel B describes all available exact model readouts and does not depend on
  tuning validity.
- Panel D uses two strictly validated exemplars. Panels E--H use the explicitly
  declared all-725 analysis population; failed strict-validation units remain
  visible in that population and may not be described as validated. Internal
  core stages in H are shared across readouts.
- Passband engagement may be described as a better descriptive predictor than
  path length only when the paired within-unit bootstrap difference is above
  zero.
- An association between passband power and SSI does not make SSI a linear
  power calculation. The responses are phase-resolved outputs of the full
  nonlinear model; the power projection identifies which retinal signals
  engage the network.
- Stagewise changes in H are reported as measured. Do not claim monotonic
  growth or assign causality to a block whose crossed confidence interval
  includes zero.
