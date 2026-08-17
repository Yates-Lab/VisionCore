# M16 successor: close the reliable-signal gap without sacrificing Jacobians

## Decision from the completed M16 evaluation

M16 epoch 415 is a viable visual-core foundation, but it is not yet the final
twin. Its exhaustive 30-session validation score is 0.597272 BPS, within
0.000948 BPS (0.16%) of the mature unregularized M12 endpoint. On the matched
24-session FixRSVP gate, M16 and the published Figure-3 twin have essentially
the same raw trial-averaged correlation (0.4436 versus 0.4468), while M16 is
lower in normalized correlation (0.5457 versus 0.6356) and single-trial
variance explained (0.0169 versus 0.0224).

The strongest diagnostic is the existing Figure-3 behavior decomposition.
Ryan's visual-only prediction has median FixRSVP CCnorm 0.5429, almost exactly
M16's intact value of 0.5457. Adding Ryan's intact behavior path raises CCnorm
to 0.6351; velocity-history alone reaches 0.6382. Thus the first successor
should not replace or simply widen the M16 visual core. The primary missing
signal is a behavior-dependent residual, especially the eye-velocity-history
component.

The Figure-4 pilot supports this division of labor. M16's response-active
filters are localized, oriented, and visibly smoother than the Figure-3 twin's
maps. Its rank-4 linear-quadratic response fidelity is 0.222 versus 0.240 for
the existing Ryan pilot. Neither teacher currently passes the predeclared 0.8
low-rank surrogate gate, so low-rank fidelity is an analysis limitation rather
than a reason to discard M16.

## Recommended model: M20 behavior-residual twin

Keep the complete M16 epoch-415 predictor, frequency masks, Gaussian readouts,
and 60 x 35 x 35 native-rate input unchanged for the frozen-core diagnostic.
Augment its existing indirect pre-readout behavior modulation with an explicit
feed-forward residual at each dataset's neural output:

```text
h = MLP(behavior_42)                         # shared 42 -> 64 -> 32 encoder
z_m16 = gaussian_readout(smooth_core(x), existing_behavior_path)
gain_i = 0.5 * tanh(W_gain_i h)              # dataset/unit-specific, zero initialized
offset_i = W_offset_i h                      # dataset/unit-specific, zero initialized
z_i = (1 + gain_i) * z_m16_i + offset_i
rate_i = softplus(z_i)
```

This remains nonrecurrent. The 40 eye-velocity-history covariates already span
roughly 30--200 ms, so temporal memory is explicit in the input rather than
hidden in a ConvGRU. The head supplies direct neuron-specific gain and offset
terms that the current shared FiLM-plus-Gaussian path can only express
indirectly. At fixed behavior, it can rescale a neuron's stimulus Jacobian but
cannot introduce new spatial frequencies or fragment its receptive-field
shape.

Use a shared behavior encoder with separate low-rank gain and offset projections
for every session. Initialize the residual exactly to zero, use weight decay on
the small new head, and retain the current bounded gain. Start with velocity
history plus eye position, but report velocity-only and position-only
counterfactuals because Ryan's cache says velocity history carries essentially
all of the useful behavior signal.

## Training sequence

1. **Frozen-core diagnostic.** Load M16 epoch 415, freeze every inherited
   parameter (including its current behavior path and Gaussian readouts), and
   train only the new behavior encoder/projections for
   24--40 epochs. This is cheap and guarantees identical stimulus-Jacobian
   shapes. Compare additive-only and gain-plus-additive heads in parallel.
2. **Restricted joint refinement.** For the better head, unfreeze the Gaussian
   readouts and final spatial stage for 32--64 epochs. Use a 10x smaller learning
   rate for existing weights than for the new head. Only unfreeze the full core
   if validation BPS remains below M12.
3. **Behavior-residual distillation fallback.** If spike-only fitting does not
   recover the CCnorm gap, fit the new head on Ryan's behavior residual on the
   training split only: `Ryan(intact) - Ryan(behavior-zeroed)`. Keep the M16
   visual output and spike loss in the objective. This transfers the smooth
   teacher-independent behavior signal without asking M16 to imitate Ryan's
   high-frequency visual Jacobian.
4. **One expensive FixRSVP gate.** Run the full matched evaluation only after a
   candidate beats 0.59822 exhaustive validation BPS and passes the fixed
   Jacobian screen. FixRSVP remains completely absent from fitting and model
   selection before this gate.

## Second intervention: restore the surround aperture

If the explicit behavior residual closes CCnorm but leaves raw correlation or
variance explained below Ryan, train the same model on 51 x 51 crops with a
13 x 13 multiscale scaffold. The 51/13 pixel-to-scaffold ratio matches M16's
35/9 ratio, so this expands the visual surround without sacrificing scaffold
resolution or adding visual parameters. Keep the 60-frame, 240-Hz support and
the first-layer anti-alias mask. Use batch 64 with two-step accumulation to
retain the current effective batch size.

This branch can be warm-started rather than trained from scratch. All learned
tensor shapes are unchanged when the input/scaffold geometry moves from 35/9
to 51/13, and the two ratios are 3.889 versus 3.923 image pixels per scaffold
unit. Consequently the learned convolution kernels, behavior paths, Gaussian
readouts, and M20 residual can all be loaded with exact shape and CID checks.
The prepared M21 config keeps the mature M16 Laplacian strengths active during
a low-learning-rate joint refinement.

This aperture experiment comes after the behavior head because the present raw
PSTH match already argues that the 35-pixel visual core is broadly adequate,
whereas the behavior decomposition identifies a quantitative missing component.

## Regularization change for any from-scratch successor

Do not optimize an always-on Laplacian penalty toward zero roughness. Treat
smoothness as a constraint with a budget:

- retain the exact first-layer temporal/spatial frequency mask permanently;
- use the current full M16 Laplacian strength through representation formation
  (approximately epochs 4--64);
- then use a hinge or adaptive-Lagrange penalty that is zero while normalized
  filter roughness is below an M16-derived ceiling and increases only when the
  ceiling is violated;
- preserve checkpoints only if the periodic fixed-context Jacobian audit also
  passes.

This lets the model spend the available smoothness budget on predictive
features instead of being continuously pushed toward unnaturally flat kernels.
M17 remains useful as the empirical half-strength control, but a constrained
schedule is preferable to choosing one fixed coefficient for all 488 epochs.

## Predeclared gates

| Quantity | Baseline | Required | Stretch target |
|---|---:|---:|---:|
| Exhaustive 30-session validation BPS | M12 0.59822 | >= 0.59822 | approach Ryan's 0.6222 checkpoint score |
| FixRSVP median raw PSTH correlation | Ryan 0.44685 | >= 0.44685 | > 0.4590 (M12) |
| FixRSVP median CCnorm | Ryan 0.63557 | >= 0.63557 | > 0.65 |
| FixRSVP median single-trial variance explained | Ryan 0.02236 | >= 0.02236 | > 0.025 |
| Exemplar temporal power >=60 Hz | M16 max 0.0123 | <= 0.02 | <= 0.015 |
| Exemplar spatial power >=0.25 cycles/pixel | M16 max 0.0114 | <= 0.03 | <= 0.015 |

The Jacobian limits are deliberately budgets rather than targets. A candidate
may be somewhat sharper than M16 and still be far cleaner than Ryan, provided
its maps remain localized, stable across contexts, and free of checkerboard or
Nyquist-edge structure. Figure-4 low-rank response fidelity is reported but is
not a model-selection gate until the same analysis passes its 0.8 gate on an
established teacher.

## Minimal experiment matrix

| Run | Core | Behavior head | Purpose |
|---|---|---|---|
| M20a | frozen M16 e415 | additive residual | identify purely additive missing behavior |
| M20b | frozen M16 e415 | bounded gain + additive residual | recommended primary run |
| M20c | frozen M16 e415 | M20b + Ryan behavior-residual distillation | fallback if noisy spike supervision underfits behavior |
| M21a | 51 px / 13 scaffold, constrained smoothness | M20b head | test missing surround after behavior is fixed |
| M21b | same as M21a | M20b + residual distillation | final high-accuracy candidate if needed |

Launch M20b first on the free GPU while M17 continues as the smoothness-dose
reference; run the additive control on the same GPU immediately afterward.
Their visual Jacobians are fixed by construction, so initial selection can be
based on exhaustive validation, behavior permutation/zeroing, and a one-session
FixRSVP smoke test rather than making M17 the only route to the next model.

## Implementation and launch audit

The optional `output_modulator` path is backward-compatible: configs without
it construct the historical model and state dictionary unchanged. Compatible
checkpoint loading verifies saved neuron identities and order, loads every M16
tensor except the new head, preserves pretrained readout biases at fit start,
and can freeze every loaded parameter. A real-checkpoint audit gave exactly
zero prediction difference between M16 and the initialized M20b model. M20b
has 5,070,028 frozen inherited parameters and 188,972 trainable output-head
parameters. Its W&B run is `zab45qp7`.

The complete-checkpoint scope and selective-unfreezing path are also tested for
the later joint-refinement stages. An M21 construction audit loaded all 296
state tensors from an interim M20b checkpoint into the 51/13 geometry and
produced finite outputs with the expected unit count.

## M20b result and promoted successors

M20b completed all 40 epochs. The selected epoch-35 checkpoint improves exact,
exhaustive validation BPS from the inherited M16 value of **0.597272** to
**0.600893** on identical examples. The mean cell-level change is +0.003999
BPS, the median is +0.002856, and 80.8% of 2,790 cells improve. The trained
checkpoint's additive-only and gain-only ablations score 0.599723 and 0.600109,
respectively, so both mechanisms carry useful signal. Zeroing only the new
residual behavior nearly restores the inherited score (0.597582), while a
within-batch behavior shift lowers it to 0.594379. This is evidence for aligned
behavioral information rather than an unconditioned recalibration.

The output gain has standard deviation 0.0223 and only 2.1e-6 of evaluated
values lie within 10% of the configured bound; the gain constraint is not
saturating. Because every inherited tensor is frozen, M20b retains M16's
stimulus-Jacobian shapes exactly at fixed behavior, up to the intended scalar
output gain.

The predeclared one-session FixRSVP smoke is effectively neutral relative to
M16: median CCnorm is 0.57132 versus 0.57148 and median model variance explained
is 0.02043 versus 0.02036. The full 24-session gate is therefore deferred until
after the short successor branches, rather than repeated for every incremental
checkpoint.

M20a is the independently optimized additive-only control. M22a is now
predeclared as the conservative accuracy successor: it warm-starts complete
M20b, freezes the visual core and original feature-space behavior path, and
co-adapts only Gaussian readouts plus the output residual for 64 epochs. This
preserves the smooth learned representation while allowing the neuron-specific
readout to allocate visual and behavioral drive jointly. M21a remains the
broader surround test. A zero-padded 35-to-51 construction audit gives output
correlation 0.9943 before any refinement; the modest scale shift is expected
because GroupNorm statistics include the larger spatial field.

## Behavior-grid audit and M24

A later preprocessing audit found a targeted difference between M16 and the
Figure-3 twin that was hidden by the common 42-dimensional tensor shape. The
Figure-3 loader first average-pools raw eye position from 240 to 120 Hz and
then computes velocity, max-normalization, symlog, and the 50-bin raised-cosine
basis. M16 retains the native time axis and computes that same operation chain
at 240 Hz. Because the historical basis implementation uses a fixed 5-ms
coordinate rather than deriving it from the dataset sampling rate, these are
not interchangeable covariates: they have different velocities and effective
history time constants.

M24 tests this without perturbing the visual representation or M16's inherited
feature-space behavior path. Its dataset exposes both the original native-rate
`behavior` tensor and a separate `output_behavior` tensor formed by applying
Ryan's exact downsample-then-transform chain and repeating the result back to
the native endpoints. Only the zero-initialized neuron-specific output
residual sees `output_behavior`; every inherited M16 tensor is frozen. A
synthetic exact-equivalence test and a real-session construction audit verify
that the odd supervision endpoints equal the legacy lower-rate transform
bit-for-bit and retain shapes `(42,)` for both behavior routes.
