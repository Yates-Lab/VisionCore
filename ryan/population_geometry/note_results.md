# Population geometry of fixational drift in an in-silico foveal V1 — results so far

Question (from `CONVERSATION.md`): how do fixational eye movements (drift) interact
with V1 population codes? Real recordings can't give many repeated fixations across
many image patches for a representative population sampling one RF location (few
repeats, small simultaneous N, single column). So we build a **representative
in-silico population** from the fig4 digital twin and do the experiment in silico.

Framing (important, see [[project_popgeom_drift_manifold_framing]]): the object of
interest is each image's **drift manifold** (population responses as the eye drifts
over that image). "Content" = between-manifold separation; "drift" = within-manifold
spread. Static presentation is out-of-domain (the twin never saw static input) and is
**not** used. Spiking noise is Poisson and independent across neurons, so population
redundancy averages it out — we work at the **rate** level.

## Methods

- **Model:** fig4 digital twin (`epoch=374, val_bps=0.6395`), a ResNet core + ConvGRU
  + Gaussian readout, per-session readout heads.
- **Population** (`01`, `02`): 288 units / 24 sessions selected by isolation
  (contamination <10%, min-rejectable RPV), twin fidelity (ccnorm >0.5), and rate
  (>1 Hz). Each unit's learned feature weights + Gaussian width are re-centered
  (`mean→0`) at the ROI center, instantiating its tuning at one common foveal
  location. Co-centering is benign: native readout positions are already tight
  (displacement median 0.42 feature-grid cells; the ROI was RF-centered at training),
  and re-centering leaves responses essentially unchanged (r = 0.986). Feature map 64×64.
- **Drift** (`_drift.py`, `03`): 2-D Brownian, MSD_2d = 4κτ (generator validated to
  0.3%). κ measured from real fixRSVP drift (radius 0.5°) with a noise-floor intercept:
  **κ ≈ 0.067 deg²/s (241 arcmin²/s)**, RMS ≈ 0.28° per 300 ms. Likely still an
  over-estimate (residual microsaccades) — flagged for calibration.
- **Behavior** (drift-consistent; `note_synthetic_behavior.md`): the twin's behavior
  input (eye_vel op-chain + raw eye_pos) is reconstructed for synthetic drift, with the
  `maxnorm` scale imputed from pooled real traces. Alignment verified (eye_pos corr =
  1.0). Behavior is non-trivial here (drift-consistent vs zeroed rate corr = 0.78,
  driven by the eye_pos channel), so it is fed, not zeroed.
- **Stimuli** (`_backimage.py`): natural backimage patches, rendered and resampled to
  37.5 ppd, 540-px full-field crops that the gaze-shift → 151-px ROI pipeline drifts over.

## Results

**Step 1 — population is well-built and representative.**
- Co-centering benign (above). All units responsive.
- Tuning (static gratings — coarse/OOD-caveated): orientation covers all angles;
  SF mode ~12 c/d (correct for fovea); OSI median 0.54; but **complex-dominated**
  (phase-modulation median 0.12). The complex bias bears on any gradient/tangent
  (simple-cell) reading of drift and is an open caveat.
- **Drift is large relative to foveal RFs** — RMS 0.14–0.28° sweeps ~2–4 RF periods
  (period ~0.08° at 12 c/d) per 300 ms. This is the Rucci/Poletti high-SF-refresh
  regime, opposite to the "drift too small to matter" worry (H2 in `CONVERSATION.md`).

**Step 2 — drift-manifold geometry (`04`, `05`; 250 patches × 20 repeats × 3 drift
amplitudes).**
1. **Content is low-dimensional and gain-dominated, intrinsically (κ-independent).**
   Between-image PR ≈ 1.8 at every drift amplitude. The dominant mode is ~91%
   single-sign and ~0.5 correlated with patch luminance → a global brightness/gain
   mode carries most of the between-image signal. This reflects tiny co-centered
   foveal RFs all sampling one location (code dominated by local luminance/contrast),
   not drift blur.
2. **Drift degrades image discriminability (functional mechanism).** 250-way decode at
   300 ms falls monotonically with drift: **0.50 → 0.36 → 0.23** (RMS 0.14 → 0.20 →
   0.28°); dimensionality is unchanged, so the harm is manifold *spreading/overlap*,
   not dimension collapse. Decode rises with integration time at every drift level:
   averaging over the drift manifold recovers content (fiber-averaging / super-resolution).
3. **Drift and content share axes beyond luminance.** Within-image drift variance in
   the top-5 between-image subspace ≈ 0.60; removing the global-gain mode leaves ≈ 0.33
   (vs random ≈ 0.02). So ~half the overlap is trivial shared brightness, but a real
   residual remains — drift moves the code along spatial-pattern content directions.

**Takeaway:** FEM is not an orthogonal nuisance in this population — it rides
content-carrying directions and measurably impairs single-fixation discrimination,
and only temporal/population integration recovers the image. Drift amplitude is the
mechanistic knob.

## Artifacts

Scripts (`VisionCore/ryan/population_geometry/`): `_pop_common.py`, `_drift.py`,
`_backimage.py`, `01_unit_inventory.py`, `02_build_and_characterize.py`,
`03_drift_smoke.py`, `04_content_drift_geometry.py`, `05_geometry_sweep.py`.
Figures/data (`outputs/figures/population_geometry/`): `unit_inventory.png`,
`population_characterization.png`, `drift_smoke.png`, `content_drift_geometry.png`,
`geometry_sweep.png`, plus `unit_inventory.csv`, `population_tuning.csv`, `*.npz`.

## Open questions / next steps

- **Residual pattern geometry:** project out the global-gain mode and redo
  dimensionality/decode/overlap on the residual spatial-pattern code — the interesting
  spatial story once brightness is removed.
- **Validate gain dominance:** is it real foveal V1 or an artifact of co-centering
  tiny RFs (vary readout pooling; compare to real recorded population geometry)?
- **κ calibration:** remove residual microsaccades, per-subject κ, resolve eye_pos
  semantics (absolute gaze vs residual FEM).
- **Complex-dominance caveat** and its effect on the drift-gradient (H1) reading.
- **H1 refinement:** isolate the retinal drift orbit vs the retinal gradient tangent
  and measure true orbit dimensionality vs drift amplitude.
