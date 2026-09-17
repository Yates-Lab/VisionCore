# Scene structure, fixational eye movements, and neuronal tuning

This analysis measures how the Figure 4 twin's instantaneous population
response depends jointly on a natural scene's spatial structure and the eye
movement that brought that scene to the retina. The complete nonlinear
encoder is evaluated for every intervention. No reconstruction network is
trained, and no grating-derived filter replaces the model.

## Inputs and intervention

`prepare.py` resolves `manuscript/analysis/selected_model_bundle.json` and
checks the selected checkpoint digest against the Figure 4 replay summary.
It uses all 40 released image patches and all 200 audited 240-Hz FEM histories
(146 drift and 54 microsaccade). Event labels were assigned independently of
responses and spectra. The 40 patches come from 13 distinct full-screen
source canvases, which are the scene clusters for uncertainty estimates.

Each entire recorded luminance canvas is normalized once using its reference
patch's original Figure 4 percentile scale. A five-level binomial Gaussian
reduce/expand pyramid gives five Laplacian detail bands and one coarse
residual, all expanded to the original image resolution. The full-canvas
mean of each component is assigned to a fixed DC image. The six mean-zero
components and DC reconstruct the original. Decomposition precedes cropping.
The overlapping bands are labeled L0–L4 and coarse residual, not represented
as sharp Fourier frequency intervals.

For each band separately, its coefficient is 0.5, 0.75, 1, 1.25, or 1.5.
All other coefficients remain one. The shared reference is computed once.
Eight additional joint-band interventions (four fixed random coefficient
vectors and their opposites, offset magnitude 0.2) are reserved for checking
the local additive approximation. Thus each scene/trajectory requires 33
forward evaluations. A single global affine contrast factor, recorded in
`design.json`, reserves headroom for **all** interventions. There is no
per-probe normalization, clipping, or matching of image energy.

## Time, geometry, and population

The existing Figure 4 coordinate conversion and image sampler render all 60
native frames from each modified static scene. A constant offset is removed
from each trace so that its final eye position is zero. The latest retinal
image is therefore identical across movements for a given coefficient
vector. The model receives the complete newest-first 250-ms history and
produces **one** native 1/240-second output bin. No responses from earlier
output times are concatenated or integrated.

This endpoint alignment is deliberate: the original Figure 4 temporal-average
SSI assay instead uses a mean-first-layer-peak-lag stabilization anchor. This
analysis records the model's peak lag but asks about a common *current*
retinal image. A stationary zero-eye-history control is included for every
scene and intervention. All behavior inputs are held at zero, as in Figure 4.

The population is all 725 available exact `(session,cid)` readouts, including
the 580 that do not meet strict grating-tuning validation. Each contributes
one output at the central location of the Figure 4 translated readout. The
readout computation is numerically checked against the full Figure 4 map.
Different translated positions are not counted as extra independent neurons.
These are model readouts pooled across sessions, not a simultaneous recorded
population. The native spatial weights and feature weights remain intact.

## Measurements

For scene `s`, eye history `e`, band `k`, and neuron `i`, the main coefficient is

```
g[s,e,k,i] = (r[s,e,a_k=1.25,i] - r[s,e,a_k=0.75,i]) / 0.5
```

Rates and signed gains are in Hz and Hz per coefficient, respectively.
Negative gain represents suppression by strengthening that structure, and a
near-zero gain represents local insensitivity. The stored five-point curves
retain nonlinear tuning. A wider secant and a second difference measure
slope stability and curvature. Held-out joint probes test whether the
individual band gains predict finite simultaneous edits.

The population rate summary is mean squared signed gain across the fixed
neurons (reported as RMS when plotted). Opposite-signed informative responses
therefore do not cancel. Event-class comparisons give each animal equal
weight within each class. Confidence intervals resample source-canvas
clusters and source eye-movement trials within animals (187 trials for 200
histories), preserving shared resampling weights when a trial contributes
multiple histories or both event classes, never treating each neuron
as an independent experimental replicate.

Two scalar controls quantify redistribution: fit moving-condition band gains
by a nonnegative multiple of stationary gains, first using one scale for the
whole population, then allowing a separate scale for every neuron. Residual
energy measures what those gain changes cannot explain. A separate band-share
display normalizes each neuron's squared gain across bands.

Balanced three-way centering removes all main and pairwise effects from the
scene × trajectory × neuron gain tensor, separately for each band. The
remaining variance is reported descriptively. Disjoint source-trial halves
test agreement of scene-specific drift/microsaccade differences across
neurons. Split ranges are not labeled as confidence intervals.

Secondary information metrics assume independent Poisson output counts:

```
local information contribution ≈ g_i^2 / (240 * baseline_rate_i)
finite-pair distance = 0.5 * sum_i (sqrt(rate_plus_i/240) - sqrt(rate_minus_i/240))^2
```

The first uses a finite-difference derivative. The second is the exact
negative log Bhattacharyya affinity between the coefficient-0.75 and
coefficient-1.25 response distributions under that noise model. A fixed-spike
analysis normalizes the response vector to probabilities and measures changes
in spike allocation. These do not estimate the empirical joint noise of the
recorded population.

## Run

Run from the repository root in the `yatesfv` environment:

```bash
conda run --no-capture-output -n yatesfv python -m unittest jake.fem_band_tuning.test_analysis -v
conda run --no-capture-output -n yatesfv python -m jake.fem_band_tuning.prepare --out-dir outputs/fem_band_tuning_20260914
conda run --no-capture-output -n yatesfv python -m jake.fem_band_tuning.inspect_inputs --out-dir outputs/fem_band_tuning_20260914
conda run --no-capture-output -n yatesfv python -m jake.fem_band_tuning.replay --out-dir outputs/fem_band_tuning_20260914 --device cuda:0 --batch-size 33 --shard 0 --shards 2
conda run --no-capture-output -n yatesfv python -m jake.fem_band_tuning.replay --out-dir outputs/fem_band_tuning_20260914 --device cuda:1 --batch-size 33 --shard 1 --shards 2
conda run --no-capture-output -n yatesfv python -m jake.fem_band_tuning.analyze --out-dir outputs/fem_band_tuning_20260914
conda run --no-capture-output -n yatesfv python -m jake.fem_band_tuning.plot --out-dir outputs/fem_band_tuning_20260914
conda run --no-capture-output -n yatesfv python -m jake.fem_band_tuning.audit --out-dir outputs/fem_band_tuning_20260914 --model
```

The two replay commands may run concurrently. Each atomically completes one
scene archive; reruns check the design hash and trace axes before reusing it.
For a smoke run add `--smoke --max-scenes 1 --max-traces 2`. Smoke artifacts
are kept separately and cannot satisfy the complete-analysis checks.

## Outputs and interpretation

`design.json` freezes inputs, checkpoint, geometry, normalization, and the
intervention matrix. `responses/scene_*.npz` contains every predicted rate.
`analysis_arrays.npz` and CSV tables contain neuron gains, full tuning curves,
scene/trace summaries, and independently measured SF/TF metadata.
`summary.json`, `FINDINGS.md`, and `audit.json` give the measured findings and
verification. Figures include actual input image variants, neuron gain maps,
all-trajectory and all-scene summaries, and a complete scene atlas.

The input review identifies two source patches (17 and 23) whose swept
151-pixel field includes screen-edge padding. They remain in the full Figure 4
source replay, but a separate 38-scene interior control repeats the population
gain comparison and scalar-gain controls. Illustrative main-panel selection
is restricted to interior scenes. Every modeled field remains more than 157
pixels inside the 540-pixel crop boundary.

The tested claim is that FEM history changes which spatial structures drive
which neurons, and which band-strength distinctions the modeled population
can support. An attention-like reweighting is a functional interpretation;
this experiment does not manipulate attention or establish an attentional
mechanism. Event-class contrasts also reflect amplitude, speed, direction,
and timing. The analysis measures existing image-band amplitudes, not the
recovery of arbitrary image features or phase. A three-way interaction alone
does not prove that nonlinear computation is necessary; the responses here
are those of the actual nonlinear twin, and curvature/joint probes separately
characterize departures from its local additive approximation.
