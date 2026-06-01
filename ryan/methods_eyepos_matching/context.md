# context.md — get up to speed in this folder

This folder (`VisionCore/ryan/methods_eyepos_matching/`) holds a
self-contained methodological note extending McFarland & Butts (2016) for
two assumption violations in `fixRSVP`:

- **(A1)** uniform trials-per-time-bin — fixRSVP has variable fixation
  durations, so n_t decays across analysis time bins.
- **(A2)** statistically stationary stimulus — fixRSVP is windowed and
  non-stationary in absolute eye position.

The methodology is grounded in **`fem-v1-fovea/references/Mcfarland-Butts-2016.pdf`**
(Eqs. 4, 6, 8, 9, 10, 13, 14, 16; the p.6228 homogeneity caveat is the
sentence the §4.5 reframe targets).

## Status

- All 15 tests pass (`uv run --with pytest pytest test_estimators.py -q`,
  ~10 min after the M6/split-half switch — the `direction2_is_more_stable`
  test runs at 12 seeds instead of 6 because M6's removal of bootstrap
  noise makes the residual stability gap small).
- Writeup builds cleanly to a self-contained HTML (`pandoc writeup.md -s
  --mathml --self-contained -o writeup.html`).
- Extension 1 is **already in production** (`VisionCore/covariance.py`:
  `estimate_rate_covariance`, `bagged_split_half_psth_covariance` with
  `weighting='pair_count'`).
- Extension 2 is implemented here and **gated**: the pipeline change to
  `covariance.py` plus the expensive GPU `fig2_decomposition` cache regen
  is NOT yet done; given the small population 1-α effect (Δ = −0.022),
  confirm wanted before touching the shared pipeline.

## File map

| file | role |
|---|---|
| `synthetic.py` | Unified rate-field generator + closed-form / MC ground truth. |
| `estimators.py` | `decompose(target=…)` — the matched LOTC estimator (Direction 1 / 2 / naive). |
| `test_estimators.py` | 15 tests: 11 Ext-2 tests + 3 sanity-check tests + Appendix §A.6 T-floor test. |
| `_style.py` | Shared matplotlib style + `figures/` save helper. |
| `fig_model.py` | Visual schematic of the unified generative model components (eye dist, GP field, masks, envelope, resulting rate). Inserted at top of writeup §2.3. |
| `fig_mechanism.py` | Geometric origin of p vs p² mismatch (Fig. 2). |
| `fig_sanity_check.py` | McFarland recovers analytical 1-α^p under (A1)+(A2) (Fig. 0). |
| `fig_time_bin_weighting.py` | Ext-1 validation on the unified synthetic (Fig. 1). |
| `fig_consistency.py` | Appendix §A.6: parallel sweep over (N, T); SEM heatmap, clipping bias, T-floor. |
| `consistency_sweep.npz` | Cached sweep results for fig_consistency (not committed). |
| `fig_naive_failure.py` | Naive vs matched on three quantities (Fig. 3). |
| `fig_correction.py` | Recovery + Direction-1/2 tradeoff + gap (Fig. 4). |
| `generate_realdata.py` | Cache-only real-data driver (do NOT recompute). |
| `realdata_results.pkl` | 397-cell cache; reused as-is by `fig_realdata.png` reference. |
| `figures/` | All generated PNGs. |
| `writeup.md` | Source. |
| `writeup.html` | Build output (committed; pandoc --mathml --self-contained). |

## Unified generative model

For neuron c, analysis time bin t, eye position e:

    r_c(t, e) = mu_0 + M_c(e) * alpha(t) * s_t(e)

- **s_t(.)** — per-time-bin i.i.d. draw of a stationary 2-D zero-mean Gaussian
  random field with covariance `K(δ) = tau^2 * exp(-||δ||^2 / (2 ell^2))`.
  (Idealization — see writeup §2.3 caveat: real fixRSVP has multiple analysis
  bins inside a single 20 Hz stimulus frame, which share the rate draw.)
- **alpha(t)** — per-time-bin amplitude (default 1; envelope demo for Ext-1
  uses a decaying alpha correlated with n_t).
- **M_c(e) ∈ [0, 1]** — spatial mask (the (A2) switch):
  - `flat`: M ≡ 1; (A2) holds.
  - `central`: exp(-||e||² / (2 ell_M²)); (A2) violated, response peaks
    at fixation (the windowing mechanism).
  - `eccentric`: 1 - exp(-||e||² / (2 ell_M²)); the bounded complement.
  - `linear`: ½(1 + tanh(x / ell_M)); smooth x-gradient.
- **mu_0** — baseline rate (default 6); marginal `r ~ N(mu_0, M(e)² τ²)`,
  Pr[r < 0] ~ 1e-9 with default params.
- **n_trials_per_time_bin** — array of length n_time_bins; variable → breaks (A1).

Eye distribution: `e ~ p = N(0, sigma^2 I)`, `sigma = 0.15°`. Close-pair
density `p² = N(0, sigma²/2 · I)` exactly.

## Closed-form decomposition

For time-bin weighting `w_t` and eye distribution `D`:

    Var_total^{D,w} = E_w[alpha^2] * tau^2 * E_D[M^2]
    Var_PSTH^{D,w}  = E_w[alpha^2] * I_{M,K,D}

where `I_{M,K,D} = ∫∫ M(e1) M(e2) K(e1 - e2) D(e1) D(e2) de1 de2`.

The ratio **1-α^{D,w} = 1 - I_{M,K,D} / (τ² · E_D[M²])** is invariant
under time-bin weighting (the envelope cancels). The Ext-1 bias is in the
*estimator*, not the truth.

For the `flat` mask + Gaussian D, K:

    1-α^p   = 2σ² / (ℓ² + 2σ²)        (the analytical sanity-check target)
    1-α^p²  = σ²  / (ℓ² + σ²)
    gap     = σ² ℓ² / [(ℓ² + 2σ²)(ℓ² + σ²)]

The gap is **non-zero under (A2)**, vanishing only as ℓ → 0 (decorrelated
rates, all FEM) or ℓ → ∞ (uniform field, no FEM); maximum ≈ 0.17 at
ℓ/σ ≈ 1.18. **This invalidates the prior writeup claim "gap = 0 iff (A2)".**
The gap measures rate spatial structure on the fixation scale, not (A2)
violation.

For `central` (Gaussian) the integrals close in closed form (Appendix
§A.5); for `eccentric`, `linear` MC at 4M samples (sampling noise ≲ 1e-3).

## The estimator (`estimators.decompose`)

Three targets:

- `naive` — McFarland's original: close-pair 2nd moment over `p²`,
  `Ȳ` over `p`. Inconsistent mix; not a variance under any single D.
- `full` (Direction 1) — q = p, the actual viewing distribution. Close-pair
  weight `1/p̂(e)` (unbounded; noisy in periphery).
- `central` (Direction 2) — q = p², the close-pair distribution. Close-pair
  weight 1; per-sample weight `∝ p̂(e)`. Bounded, stable.

Both consistent targets restore term-by-term LOTC consistency. Their gap
is the fixation-scale spatial-structure measure.

`time_bin_weighting` ∈ {`pair_count`, `uniform`} — the Ext-1 axis. Under
constant n_t the two coincide; under variable n_t with envelope correlated
to n_t they differ, and only `pair_count` matches `Crate`'s intrinsic
weighting.

`cpsth_method` ∈ {`mcfarland`, `split_half`} — how Cpsth is debiased
against same-time-bin observation noise (Poisson + simultaneous cross-cell
noise correlations).

- `mcfarland` (default): McFarland Eq. 6/M12 all-distinct-pair second
  moment, with target-importance per-pair weights `w_i*w_j` (where
  `w_i = q(e_i)/p(e_i)`) and the same across-bin scheme as Crate. The
  close-pair Crate (Eq. 8) and unconditioned Cpsth (Eq. 6) are then the
  same estimator at the two ends of the Δe axis. `_all_pairs_second_moment`
  uses the algebraic identity `2 Σ_{i<j} w_i w_j S_i⊗S_j = (Σw_i S_i)⊗(Σw_i S_i) − Σ w_i² S_i⊗S_i`
  so per-bin compute is O(n_t·C²), no explicit pair tensor.
- `split_half`: bagged split-half PSTH covariance (n_boot=20). Stochastic;
  same population target in expectation; converges to `mcfarland` as
  n_boot → ∞. Retained in `_split_half_psth_cov` for the production
  pipeline's parallel implementation and as a fallback.

Writeup §A.7 motivates the choice (M6's conceptual unification with
Crate); the production pipeline (`VisionCore/covariance.py`,
`bagged_split_half_psth_covariance`) still uses split-half, and the two
estimators agree within bootstrap noise at our (N, T, C) scales.

## Key empirical findings

- **Sanity check.** `decompose(target='naive', time_bin_weighting='pair_count')`
  with constant n_t recovers analytical `1-α^p` across an ell/σ sweep
  covering (0,1) — Fig. 0A.
- **Naive bias (synthetic).** On `central`-mask cells: naive over-states
  `1-α` (close pairs over-represent the center where M² is large). On
  `eccentric` masks: under-states. Matched recovers truth.
- **Naive leak (synthetic).** Noise correlation and Fano leak are MUCH
  milder under the unified (multiplicative-mask) model than the prior
  additive model predicted: med |r| 0.015 vs 0.010 matched; Fano med
  ≈ 1.01 for both. The mean of r stays at mu_0 (e-independent), so the
  cross-term that dominated the additive model vanishes.
- **Real data (397 good cells, cached).** Naive median 1-α 0.734 reproduces
  fig2's 0.732; Direction 1 0.702 (population bias −0.022); Direction 2
  0.608; gap median 0.089; Fano 0.846 → 0.875.
- **Gap reframe (§4.5).** Gap = 0.089 in real data means the cell rate has
  spatial structure on the fixation scale (expected for any cell with a
  finite RF), NOT (A2) violation. The (A2) baseline for the random field
  alone at ℓ = σ is ≈ 0.17.

## Open items (active)

1. **Appendix §A.6 — DONE.** Explored SEM(N, T) via the parallel
   `fig_consistency.py` sweep + closed-form derivation. The across-time-bin
   floor `sd[1-α̂] = α√(2/(T−1))` is derived and matches empirics; the [0,1]
   clipping bias is characterised against the truncated-Gaussian formula.

2. **Terminology — RESOLVED.** "phase" (the McFarland term, overloaded with
   the fixRSVP 20 Hz stimulus frame) has been globally renamed to
   `time_bin` (analysis time bin, the 60/120 Hz unit the pipeline iterates
   over). Writeup §2.3 carries the caveat that the synthetic models each
   bin as i.i.d. — a stimulus-frame idealization — so the effective T for
   the floor in real data is closer to `(#fixations × #stim-frames-per-fixation)`
   than the raw bin count. The within-stimulus-frame reliability question is
   a future direction.

3. **Production pipeline change for Ext-2 (gated).** Add target-distribution
   weights to `estimate_rate_covariance` / `bagged_split_half_psth_covariance`
   / the Ctotal computation in `VisionCore/covariance.py`. Default
   `target='naive'` so current numbers stand. Then the expensive GPU
   `fig2_decomposition` cache regen. Confirm wanted before touching shared
   pipeline.

## Build / test commands

Run from this folder. Workspace `.venv` is at the v1-fovea repo root;
`uv run` resolves it automatically.

    # Generator self-check
    uv run python synthetic.py

    # Tests (15 tests, ~7 minutes -- random field Cholesky per time bin)
    uv run --with pytest pytest test_estimators.py -q

    # Figures
    uv run python fig_model.py
    uv run python fig_mechanism.py
    uv run python fig_sanity_check.py
    uv run python fig_time_bin_weighting.py
    uv run python fig_naive_failure.py
    uv run python fig_correction.py
    uv run python fig_consistency.py            # parallel sweep, cached

    # Writeup
    pandoc writeup.md -s --mathml --self-contained -o writeup.html

## External pointers

- McFarland & Butts 2016 paper:
  `fem-v1-fovea/references/Mcfarland-Butts-2016.pdf`.
- Production pipeline: `VisionCore/VisionCore/covariance.py`
  (`pipeline_one_minus_alpha`, `estimate_rate_covariance`,
  `bagged_split_half_psth_covariance`).
- Related figure-2 bias diagnosis:
  `VisionCore/ryan/fig2/bias_diagnosis/FINAL_REPORT.md`.
- Auto-memory note:
  `~/.claude/projects/-home-ryanress-v1-fovea/memory/project_eyepos_distribution_matching.md`.
- Related panel-D resolution memory:
  `~/.claude/projects/-home-ryanress-v1-fovea/memory/project_fig4_panelD_one_minus_alpha.md`.

## Constraints (durable)

- Python only via `uv run`. Workspace `.venv` at the repo root.
- Do NOT touch `VisionCore/covariance.py` or regenerate the
  `fig2_decomposition` GPU cache without explicit user approval.
- Do NOT regenerate `realdata_results.pkl`; reuse the cached numbers.
- TDD for new generator changes (failing test first, then implementation).
- Single-line commit messages, no co-author trailer.
- Commit only when asked.
