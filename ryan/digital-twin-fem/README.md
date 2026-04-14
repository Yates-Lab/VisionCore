# Digital twin FEM analyses

In-silico analyses supporting the final section of the foveal V1 manuscript. Uses the figure-3 digital twin to test whether fixational eye movements (FEMs) are necessary for foveal V1 to support fine visual discrimination.

## Research question

**Do FEMs convert the spatial code in foveal V1 into a spatiotemporal code that supports fine discrimination?**

The manuscript establishes through real data that (fig 2) FEMs dominate shared rate variability in foveal V1, and (fig 3) the digital twin captures this FEM-driven variability. Both are structural claims about the code. The final section asks the functional question: *what is this FEM-driven variability for?*

The hypothesis is the Rucci/Victor space-to-time line — FEMs scan local features across each neuron's receptive field, turning a static spatial pattern into a stimulus-dependent temporal sequence on each neuron. Recurrent dynamics (adaptation) cause a static image to lose information with integration time; FEMs prevent that loss and let discriminability accumulate.

## Why a digital-twin analysis

A behavioral experiment can't cleanly manipulate FEMs in an awake animal — you can suppress them with retinal stabilization, but that disturbs other things too. The digital twin lets us:

1. Generate counterfactual retinal inputs (real FEM, no FEM, scaled FEM) for the same stimulus.
2. Pool simulated neurons across sessions by translating readout positions — the population-scale benefit of the shared core.
3. Ablate model components (specifically, recurrent memory) that stand in for adaptation, to test mechanism.

These manipulations have no experimental analog.

## Why not Fisher information

Initially considered as a second primary metric alongside a decoder. Dropped for three reasons:

1. **Temporal-independence violation.** Cumulative Fisher info assumes conditionally independent observations over time. The ConvGRU creates strong autocorrelation in the rates, and does so *asymmetrically* between FEM conditions (no-FEM is more redundant). Summing per-bin Fisher info would overstate cumulative info in both conditions but disproportionately in no-FEM, potentially flipping or exaggerating the comparison.
2. **Fictional noise model.** The digital twin was trained on rate prediction, not on Poisson residuals. Asserting Poisson noise to compute $I(\theta) = \sum_i (r_i')^2 / r_i$ is a layer of assumption the model doesn't actually satisfy.
3. **Wrong tool for the task.** Fisher info bounds local discrimination of a continuous parameter. The tumbling-E task is 4-way categorical with 90° separations. A decoder does the task directly, under a noise model we fully control.

A trained decoder on Poisson-sampled trials is more honest about what we're claiming.

## Why not ImageNet

Initially considered. Dropped because:

1. Foveal RFs cover roughly 1°; ImageNet objects are not scaled to this.
2. The encoder was trained on specific natural-image statistics — ImageNet is out-of-distribution.
3. A V1 → ImageNet-label readout requires heavy downstream machinery (V2/V4/IT equivalent) that the model doesn't have; adding one introduces confounds that dominate any V1 effect.

Tumbling-E is parametric, foveal-sized, psychophysically grounded, and matches the section-4 placeholder in `main.tex`.

## The three analyses

### A — Discrimination vs integration time

Does accuracy on the 4-way E task grow faster and reach a higher asymptote with FEMs than without? This is the core functional claim. Four FEM conditions (real, none, scaled ×0.5, ×2) probe both the presence of FEMs and whether real FEM amplitude is near-optimal.

### B — Decoder ablation

Is the FEM advantage specifically *temporal*, or does it just reflect additional independent samples of the same spatial code? Three linear decoders — instantaneous, time-averaged, flattened-temporal — tested on the same trials. If the temporal decoder benefits disproportionately from FEM relative to the time-averaged one, the FEM advantage is a genuine temporal code. If not, FEMs are "free samples" of a spatial code — weaker but still valid.

### C — Adaptation ablation

Does the FEM advantage require recurrent dynamics in the encoder? 2×2 factorial on {real FEM, no FEM} × {GRU memory on, GRU memory off}. GRU memory is disabled by zeroing its hidden state at every time step. If the FEM advantage collapses without recurrence, the proposed mechanism (adaptation prevents info accumulation → FEMs counteract it) is supported. If the advantage persists without recurrence, the mechanism is elsewhere and the story needs rethinking.

## Key design choices

- **Population.** Simulated neurons are drawn from (real CCmax-reliable units) × (foveal readout grid positions), pooled across sessions, randomly subsampled to $N$. $N$ is a hyperparameter swept as a diagnostic in analysis A to avoid running in a ceiling regime.
- **Stimulus.** Tumbling-E optotype at a fixed foveal-centered size (~0.3° default). Size sweep for an acuity-threshold curve is a TODO.
- **Trial generation.** Poisson-sample spike counts from the model's rate output, trial by trial. Gives us a noise model we control, independent of the model's training loss.
- **Decoder.** Multinomial logistic regression with matched L2 across decoder classes. GRU-based decoder is a TODO.
- **Adaptation ablation mechanism.** Hidden-state reset at every step; same weights, same architecture, memory removed. A retrained no-recurrence model would be stronger — listed as TODO.
- **Scope.** Within-fixation. Saccade-crossing analyses are a TODO.
- **Organization.** Three standalone scripts plus a shared `_common.py`. No figure composition; that belongs to a separate `generate_figure4.py`.

## What success looks like

Analysis A: accuracy($T$) rises faster and higher under real FEM than no FEM, with scaled conditions bracketing real such that real is near the peak of the dose-response.

Analysis B: $\Delta_{\text{temporal}} > \Delta_{\text{time-averaged}}$ for the real-vs-none FEM contrast.

Analysis C: $\Delta_{\text{FEM}}$ large with GRU memory on, small or absent with GRU memory off.

Any of these failing is informative. Analysis A failing means the whole story needs rethinking. B failing weakens the "spatial-to-temporal" framing to "FEMs give free samples." C failing means adaptation is not the mechanism and we need a different one.

## References

- Plan: `plans/2026-04-13-digital-twin-fem-design.md`
- Model loading: `VisionCore/ryan/fig3/generate_figure3.py`
- Existing spatial-info infrastructure: `VisionCore/scripts/spatial_info.py`, `VisionCore/scripts/fixrsvp_digitaltwin_spatialinfo.py`
- Section-4 placeholder: `fem-v1-fovea/main.tex` §4.5
