# Panel D (model vs empirical 1−α) — investigation handoff

**Status as of 2026-05-28.** Panel D is built and validated *at the median*, but a
per-cell discrepancy is unresolved and the decisive control is deferred to a fresh
session for an **independent audit**. This note is deliberately factual: treat the
interpretations below as **open hypotheses to test**, not settled conclusions.

## What panel D asks

1−α = fraction of a cell's rate variance driven by fixational eye movements (FEM)
vs. the stimulus-locked (PSTH) fraction α. Panel D compares the digital twin's
1−α (from its predicted rates) against the empirically measured 1−α (Fig 2), per
cell, both subjects. It is a test of whether the twin captures FEM effects through
the *retinal* consequence of drift (stimulus shifting on the retina).

## What was built

- **Estimator** `VisionCore/covariance.py::rate_variance_components` — analytic
  one-way random-effects ANOVA of the deterministic model rate grouped by stimulus
  phase. Returns `sigma2_within` (FEM), `sigma2_between` (PSTH, debiased via
  `(MS_between − MS_within)/n0`), `sigma2_total`, `one_minus_alpha`. Plus
  `psth_variance_splithalf` (split-half cross-check). Full derivation in the
  docstring. Rationale: the twin is deterministic, so there is no Poisson noise to
  remove and the empirical eye-distance + split-half machinery collapses to a plain
  variance-components split.
- **Tests** `VisionCore/tests/test_covariance.py` — 6 TDD tests (unbiased recovery,
  affine invariance, pure-PSTH→0, pure-FEM→1 with debiasing exposed, exact unbalanced
  n0, split-half agreement). All 17 covariance tests pass.
- **Panel** `ryan/fig4/generate_fig4d.py::plot_panel_d` — scatter of model vs
  empirical 1−α; analytic primary, split-half reported as agreement check. Full
  logic in the module docstring. NOT yet integrated into `generate_figure4.py`.
- **Diagnostics** `ryan/fig4/diagnose_fig4d_alpha.py` (cache-only) and
  `ryan/fig4/simulate_fig4d_control.py` (Poisson-simulation pipeline control —
  PRELIMINARY/confounded, see below).

## Cache change (already applied + verified)

`_fig4_data.py::_run_inference` now stores per session `eyepos_used` (n_trials,
n_time, 2) and `valid_mask` (eye-finite bins); the two `ccnorm_split_half` calls
are seeded (rng=42/43) for reproducibility. Cache regenerated. Verified vs backup
(`outputs/cache/fig3_digitaltwin.pkl.bak`): deterministic fields bit-identical
(max|diff|=0.0), only 3/1703 good-threshold flips (395→398), eyepos finite under
valid_mask.

## Empirical findings

- **Median agreement (good):** model-analytic ≈ empirical. Allen median emp 0.752 /
  model 0.730; Logan 0.588 / 0.657; All 0.732 / 0.717. Session 0: model-analytic
  0.724 vs fig2 0.725. The twin recapitulates 1−α at the population level.
- **Per-cell scatter is moderate** (Spearman ρ ≈ 0.41 Allen / 0.44 All), with an
  **upper-left cloud**: 34/398 good cells at emp<0.5 & model>0.6. NOT a clipping
  artifact — `sigma2_between` clips to 0 for 0% of cells.
- **Logan weak correlation** (ρ≈0.21, N=49): substantially range restriction
  (empirical 1−α tops out at 0.77 vs Allen's full 0–1 span) plus a small (~0.07)
  upward model bias.
- **Analytic vs split-half:** r=0.984, median|diff|=0.009 — the two model-side
  debiasings agree, so the analytic estimator is internally sound.

## The open problem (the reviewer risk)

Panel D as drawn invites the critique: *the model is more sensitive to eye position
than the real neurons, so it may not be an adequate digital homologue.* We need to
either confirm that (and understand it) or show it is an artifact of comparing
**two different estimators** (clean ANOVA on model rates vs. the full empirical
pipeline on noisy spikes).

### Preliminary control + its confound

`simulate_fig4d_control.py` Poisson-samples the model rates and runs the *identical*
`run_covariance_decomposition` (1-bin, below_threshold 0.05°). On **fig4-aligned**
data this does **not** reproduce fig2 (obs_pipeline ~0.476 vs fig2 ~0.725), so it is
confounded — the fig4 alignment differs from fig2's `align_fixrsvp_trials`. Localized
(session 0, sim spikes): pipeline components correlate with the model's analytic
components (PSTH r=0.74, Crate r=0.91) but are scale-inflated (PSTH ~2.7×, Crate
~1.3×); the differential PSTH inflation deflates the pipeline's 1−α.

### Leading hypothesis to test (Ryan's insight)

The below_threshold intercept estimates Crate from **pairs** of trials with eye
distance < 0.05°. Fixational eye position is ~Gaussian, so close pairs are
over-represented at **central** eye positions (squared-density integral dominated by
the center). The pipeline thus integrates rate variance over a **narrower, more
central** eye-position distribution than the ANOVA, which uses **all** samples (full
fixational distribution, including peripheral drifts that shift the stimulus further
on the retina). If FEM variance grows with eccentricity, the two estimators measure
FEM over different distributions and are **not directly comparable** — possibly
dissolving the "more sensitive" reading. **Equal-footing test:** compute the model
1−α on the same close-pair distribution the pipeline uses (or measure both over a
matched eye-position distribution).

## Next session — suggested angles (be creative, multi-angle)

1. **Faithful control:** prepare simulated model spikes in fig2's exact
   `align_fixrsvp_trials` frame so obs_pipeline reproduces fig2, *then* compare
   sim_pipeline vs model-analytic vs obs vs fig2.
2. **Eye-position-distribution matching:** test the pair-distribution hypothesis
   directly — does restricting the ANOVA to central eye positions (or weighting by
   the close-pair density) reconcile model and empirical 1−α?
3. **Eccentricity dependence:** is the model's FEM variance eccentricity-dependent?
   (Would explain estimator divergence.)
4. **Empirical reliability:** test-retest the fig2 per-cell 1−α (seeds / split) to
   quantify how much of the panel-D scatter is measurement noise.
5. **Independent audit:** re-derive the estimator comparison fresh; do not assume the
   above hypotheses are correct.

## How to run

```bash
cd VisionCore/ryan/fig4
uv run python generate_fig4d.py            # panel + per-subject stats
uv run python diagnose_fig4d_alpha.py      # cache-only floor/cloud/Logan diagnostics
uv run python simulate_fig4d_control.py --n-sim 5   # PRELIMINARY control (~slow, GPU)
uv run --with pytest pytest ../../tests/test_covariance.py   # estimator tests
```
