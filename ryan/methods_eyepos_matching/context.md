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
sentence the §4.3 reframe targets).

## Status

- All 24 estimator tests pass (`uv run --with pytest pytest test_estimators.py -q`,
  ~12 min after the M6/split-half switch). Pipeline tests in
  `test_pipeline.py` (3 tests, including a bit-identical match against the
  legacy torch `extract_windows`) pass in ~30 s.
- Writeup builds cleanly to a self-contained HTML (`pandoc writeup.md -s
  --mathml --self-contained --lua-filter=number-eqs.lua -o writeup.html`);
  the `number-eqs.lua` filter re-adds equation numbers that pandoc's MathML
  writer drops (amsmath `\tag{}`). The ANOVA side note builds with the same
  command swapping `writeup.md` for `note_anova.md`.
- Extension 1 is **already in production** (`VisionCore/covariance.py`:
  `estimate_rate_covariance`, `bagged_split_half_psth_covariance` with
  `weighting='pair_count'`).
- Extension 2: estimator validated, **a CPU-parallel reimplementation of the
  full Figure-2 pipeline lives in this folder** (now `note_pipeline.md` §7) and reproduces the legacy
  numbers at `target='naive'` within the documented tolerance; the production
  GPU swap of `VisionCore/covariance.py` + `fig2_decomposition` cache regen
  is the only remaining gated step.

## File map

| file | role |
|---|---|
| `synthetic.py` | Unified rate-field generator + closed-form ground truth (all three masks close analytically). GP field sampled via eigendecomposition (`_draw_field_at`, stable for dense/smooth covariances). `make_trajectory_session` is the trajectory-mode TEST fixture (centroid + per-bin drift); its σ_drift sweep / σ_traj `traj_truth` are deprecated (writeup no longer references them). |
| `estimators.py` | `decompose(target=…)` — single-bin §4.2 matched estimator. `decompose_trajectory(target=…, reduction=…)` — §4.4 multi-bin extension: RMS-trajectory close-pair filter + reduce each trajectory to one representative point (`_geometric_median`, default; or `centroid`) + ONE KDE on those points reusing the §4.2 weights (p² implied; no separate close-pair KDE). |
| `test_estimators.py` | 27 tests: 12 single-bin Ext-2 + 3 sanity + Appendix §A.9 T-floor + trajectory-mode tests (2 `_geometric_median` unit, flat-limit recovery, geomedian recovery at realistic drift, geomedian≈centroid in flat regime, naive-biased) + 1 `_draw_field_at` coincident-point eigendecomposition test + 2 trial_count direction tests + 1 additive-vs-multiplicative cross-cell bias (§3). |
| `_style.py` | Shared matplotlib style + `figures/` save helper. |
| `fig_model.py` | Visual schematic of the unified generative model components (eye dist, GP field, masks, envelope, resulting rate). Inserted at top of writeup §2.3. |
| `fig_mechanism.py` | Geometric origin of p vs p² mismatch + closed-form naive bias across masks (2-row Fig. 2: A/B top, C full-width flat/central/eccentric bars at σ_M=σ_e; flat ≈ no bias). Concludes writeup §4.1. |
| `fig_distribution_truth.py` | Closed-form 1-α depends on eye distribution (flat) and mask width (central). End of §2.3 (Fig. 0a). |
| `fig_sanity_check.py` | McFarland recovers analytical 1-α^p under (A1)+(A2) (Fig. 0b; panels A recovery + B threshold). |
| `fig_weighting_bias.py` | Ext-1 (Fig. 1, 3 panels A/C/D): A = variable-n_t weighting divergence; C = synthetic shuffle-null Dz (additive transient biases mixed, multiplicative gain does not; truth 0); D = real shuffle-null Dz (mixed −0.020 vs consistent −0.001). Replaces the old `fig_time_bin_weighting.py`. |
| `compute_weighting_data.py` | Ext-1 real-data driver (cache-only, all 25 sessions from `cache/aligned_sessions.pkl`): mixed vs consistent weighting -> shuffle-null Dz, corrected NC. Writes `cache/weighting_realdata.pkl`. No GPU / covariance.py / realdata_results.pkl touched. |
| `fig_consistency.py` | Appendix §A.9: parallel sweep over (N, T); SEM heatmap, clipping bias, T-floor. |
| `consistency_sweep.npz` | Cached sweep results for fig_consistency (not committed). |
| `fig_recovery.py` | Matched estimators recover analytical 1-α^p (full) / 1-α^{p²} (central) on central+eccentric across an ℓ/σ sweep; naive biased off both (Fig. 3, σ_M=σ_e, N=800). Concludes writeup §4.2. Replaced the old fig_naive_failure.py + fig_correction.py (deleted 2026-06-18). |
| `fig_trajectory.py` | §4.4 multi-bin extension validated on REAL eye traces (Fig. 4): A real within-window flatness + example windows; B ℓ/σ recovery (D1→1-α^p, D2→1-α^p², naive biased) with synthetic rates on real eye windows, truth evaluated non-parametrically over the real eye sample; C geomedian-vs-centroid divergence confined to the microsaccade tail. Replaces the old synthetic σ_drift sweep. |
| `generate_realdata.py` | Cache-only real-data driver (do NOT recompute). Single-bin close-pair filter (see §4.5 caveat re: §4.4). Saves Fig. 5 (writeup §4.5). |
| `realdata_results.pkl` | 397-cell cache; reused as-is by `fig_realdata.png` reference (now Fig. 5). |
| `figures/` | All generated PNGs. |
| `writeup.md` | Main methods writeup source. §4 = Ext-2: 4.1 naive failures (merged old 4.1–4.3, concl. Fig 2), 4.2 corrected estimator (concl. Fig 3), 4.3 Dir-1/2 tradeoff+gap, 4.4 multi-bin trajectory (Fig 4), 4.5 real data (Fig 5). Old §5 deleted; §6–§7 moved to note_pipeline.md (2026-06-18 restructure). |
| `note_pipeline.md` | Implementation note: production-pipeline state (§6) + CPU-parallel Figure-2 reimplementation/validation (§7, Figs 7–9), moved out of writeup.md. Section/figure numbers retained from the main note. Builds to note_pipeline.html. |
| `writeup.html` | Build output (committed; pandoc --mathml --self-contained). |
| `compute_closepair_density.py` | Cache-only driver (all 25 sessions): directly-estimated close-pair density `p_pair` (KDE on close-pair representative-midpoints) vs the squared marginal `p̂²`, + re-runs §4.5 decomposition under `closepair_density ∈ {squared,direct}`. Writes `cache/closepair_density.pkl`. For `note_closepair_density.md`. |
| `fig_closepair_density.py` | Figure for `note_closepair_density.md`: A `p` / `p²` / `p_pair` x-marginal overlay; B per-session close-pair variance ratio vs ideal 0.5; C/D per-cell 1-α full/central squared-vs-direct. |
| `note_closepair_density.md` | Side note: directly estimating `p_pair` vs the §A.5 `p̂²` assumption. Real close pairs are MORE central than `p²` (var ratio med 0.42 vs 0.5, KL 0.16); full 1-α robust (+0.017), central −0.060, gap 0.119→0.183, naive/Fano invariant. Builds to note_closepair_density.html. |
| `note_anova.md` | Side note: one-way ANOVA on known rates + fig4 panel D investigation. Tangential to main writeup. |
| `note_anova.html` | Build output (committed; pandoc --mathml --self-contained). |
| `fig_anova.py` | Synthetic validation for `note_anova.md` §5 (ANOVA recovers $1-\alpha^p$ across all three masks). |
| `fig_panel_d_anova.py` | Real-data panel D: cell-side matched close-pair (naive / D1 pair / D1 trial) vs twin ANOVA. |
| `fig_panel_d_closepair.py` | Real-data panel D: matched close-pair D1 on BOTH cells and twin. Figure 2 of `note_anova.md`. |
| `legacy/` | Frozen 2026-06-02 snapshot of `VisionCore/{covariance,stats,subspace}.py` + `VisionCore/ryan/fig2/compute_fig2_data.py`. Used as the §7 comparator. Do not edit. |
| `data_loading.py` | One-shot aligned-session cache builder (`cache/aligned_sessions.pkl`). Calls `prepare_data` + `align_fixrsvp_trials` + R² inclusion stats once per session. Self-contained boundary for §7. |
| `pipeline.py` | §7 per-session driver. Numpy port of `extract_windows` (bit-identical to legacy on the test fixture) + `decompose_trajectory` per (window, target) + naive eye-shuffle nulls. Also exposes `decompose_session_legacy` (the snapshot adapter for CPU comparison). |
| `metrics.py` | §7 stage-2 derived metrics (1-α / Fano / NC means), lifted from `legacy.compute_fig2_data`. Loops over targets. |
| `compute_methods_data.py` | §7 orchestrator. `joblib(loky)` parallel across sessions; `--legacy` and `--both` modes. Emits `cache/{methods,legacy}_{decomposition,derived}.pkl`. |
| `test_pipeline.py` | 3 tests: extract-windows bit-equivalence vs legacy, methods-pipeline schema, legacy-adapter smoke run. |
| `profile_pipeline.py` | cProfile a single session at window=2; writes `cache/profile.prof`. |
| `timing.py` | Per-(session, window) wall-time CSV for methods vs legacy snapshot (CPU vs CPU). Writes `cache/timing.csv`. |
| `fig_pipeline_equivalence.py` | Fig. 7 (§7.2): per-cell legacy vs methods at target='naive' scatter (Crate diag, Cpsth diag, 1-α). Exits non-zero if pass criteria fail. |
| `fig_pipeline_corrections.py` | Fig. 8 (§7.3): 3×3 grid naive/full/central × {1-α dist, Fano scatter, per-session Δz}. |
| `fig_pipeline_speed.py` | Fig. 9 (§7.4): per-session and per-window wall-time bars, methods vs legacy. |

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
  - `central`: exp(-||e||² / (2 σ_M²)); (A2) violated, response peaks
    at fixation (the windowing mechanism).
  - `eccentric`: 1 - exp(-||e||² / (2 σ_M²)); the bounded complement.

  All three masks have closed-form 1-α (writeup Appendix A.1–A.4),
  MC-verified; `ground_truth` returns the exact value (no Monte Carlo).
- **mu_0** — baseline rate (default 6); marginal `r ~ N(mu_0, M(e)² τ²)`,
  Pr[r < 0] ~ 1e-9 with default params.
- **n_trials_per_time_bin** — array of length n_time_bins; variable → breaks (A1).
  Per-cell fields are independent (true cross-covariance 0); the §3 cross-cell
  weighting bias needs no cross-cell co-tuning, so there is no shared-field knob.

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

All three masks (`flat`, `central`, `eccentric`) close in closed form
(Appendix §A.1–A.4), each MC-verified to ≲ 1e-3.

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

`closepair_density` ∈ {`direct` (default since 2026-06-24), `squared`} — how the
close-pair midpoint density (the Crate sampling density) is estimated for the
importance weights. `direct` (default): fit a separate KDE on the realized
close-pair midpoints. `squared`: assume `p_pair = p̂²` (the §A.5 Δe→0 identity;
pre-2026-06-24 behavior, bit-for-bit). `full`'s close-pair weight is
`p̂/p̂_pair` (`1/p̂` under squared); `central`'s per-sample weight is `p̂_pair/p̂`
(`p̂` under squared) — its close-pair weight stays 1. Threaded through
`decompose`, `decompose_trajectory`, and `pipeline.decompose_session`. The
synthetic closed-form validation figures/tests (fig_recovery, fig_trajectory,
fig_anova; `_seed_stats*` helpers) pin `squared` (p_pair=p² holds on Gaussian-eye
synthetic); the real-data Ext-2 drivers (generate_realdata, compute_methods_data)
use the `direct` default. See `note_closepair_density.md`.

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

Writeup §A.10 motivates the choice (M6's conceptual unification with
Crate); the production pipeline (`VisionCore/covariance.py`,
`bagged_split_half_psth_covariance`) still uses split-half, and the two
estimators agree within bootstrap noise at our (N, T, C) scales.

## Key empirical findings

- **Ext-1 cross-cell bias (§3, NEW).** Mixing time-bin weightings (close-pair
  Crate pair-count, Cpsth uniform 1/T, Ybar trial-count — the pre-fix production
  state) biases the CROSS-CELL covariance, not the per-cell 1-α. Real data (25
  sessions, `compute_weighting_data.py`): corrected NC median +0.019 (consistent)
  → +0.008 (mixed, many pairs negative); shuffle-null Dz −0.001 (consistent) →
  −0.020 (mixed). The bias is **additive, not multiplicative**: a gain envelope is
  weighting-invariant (no bias); a random per-cell additive onset transient
  reproduces it (μ_pair > μ_trial, δ_c>0 → off-diagonal Crate inflated ≈
  μ0(δ_m+δ_n)). It needs **no cross-cell co-tuning**: the bias survives with
  independent fields AND independent per-cell transient shapes (cross-cell PSTH
  cov ≈ 0), because the large μ0 multiplies each cell's own δ_c — so the old
  `field_corr`/"PSTH-covariance substrate" framing was dropped (2026-06-17). The
  two consistent weights (n_t, 1/T) are equivalent in the model; default to n_t
  (lower variance). Reproduces the `ryan/fig2/bias_diagnosis` single-session
  result at population scale.
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
- **Close-pair density `p_pair` vs `p̂²` (`note_closepair_density.md`).** The
  trajectory-mode weights assume the close-pair midpoint density = `p̂²` (§A.5,
  exact only single-bin Δe→0). Directly estimating it (KDE on close-pair
  representative-midpoints, `closepair_density='direct'`): real close pairs are
  MORE central than `p²` (population-median variance ratio 0.42 vs ideal 0.5;
  KL 0.16) — leptokurtic fixations + RMS-trajectory match keep central
  fixations. Effect on §4.5 (1359 cells): full 1-α 0.692→0.709 (+0.017,
  per-cell median +0.001 — robust); central 0.537→0.477 (−0.060); gap
  0.119→0.183; naive & Fano invariant (no importance weights). Headline (full,
  Direction 1) holds by measurement, not just the Δe→0 identity. **`direct` is
  now the estimator default (2026-06-24) and the main writeup reports it
  throughout** — §4.5 Figure 5 (realdata_results.pkl schema 4), §4.5 NC + Figure
  6 (methods bundle regenerated direct), note_pipeline §7.3 Fig 8, note_consistency
  tables all regenerated/updated to direct. Fig 7 equivalence (naive) still PASSES.
- **Gap reframe (§4.3).** Gap = 0.089 in real data means the cell rate has
  spatial structure on the fixation scale (expected for any cell with a
  finite RF), NOT (A2) violation. The (A2) baseline for the random field
  alone at ℓ = σ is ≈ 0.17.

## Open items (active)

1. **Appendix §A.9 — DONE.** Explored SEM(N, T) via the parallel
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

3. **§4.4 multi-bin trajectory extension — DONE (single-point reduction,
   real-eye validation; 2026-06-18 rework).** `decompose_trajectory` reduces
   each trajectory to one representative point (`_geometric_median`, robust to
   within-window microsaccades; `reduction='centroid'` optional) and fits ONE
   KDE on those points, reusing the §4.2 single-bin weights (p² implied by p̂;
   no separate close-pair KDE, no ℝ^{2T} density). RMS-trajectory close-pair
   filter kept. Exact in the flat-trajectory limit; the small-drift residual is
   validated **on real eye traces** (`fig_trajectory.py`): synthetic rates on
   real `fixRSVP` eye windows, truth evaluated non-parametrically over the real
   eye sample (p over representative points, p² over their close-pair
   midpoints). Real windows are nearly flat (median within-window drift
   ≈0.07σ_e); reductions diverge only in a ~16% microsaccade tail. D1 recovers
   1-α^p, D2 recovers 1-α^p² with small bias over ℓ/σ∈[0.5,1.5]; naive biased.
   The old synthetic σ_drift sweep + σ_traj closed-form truth are deprecated
   (kept in `make_trajectory_session` for tests, removed from the writeup).
   **Fixed a real bug**: `_draw_field_at` used Cholesky+growing-jitter, which
   on dense/smooth covariances injected diagonal noise that decorrelated close
   pairs and inflated 1-α; now eigendecomposition sampling (variance- and
   coincident-point-preserving). `pipeline.py`/`note_pipeline.md` §7 synced to
   the same single-point reduction.

4. **Methods-folder parallel pipeline — DONE (now `note_pipeline.md` §7).** A CPU-parallel
   reimplementation of the entire Figure-2 LOTC pipeline lives in
   `pipeline.py` / `metrics.py` / `compute_methods_data.py`, using
   `decompose_trajectory` for stage 1 and the legacy stage-2 aggregator
   (lifted into `metrics.py`). The numpy windowing port is bit-identical to
   the legacy torch implementation (`test_pipeline.py`), so equivalence at
   `target='naive'` is structural rather than empirical. Validated end-to-end
   via `fig_pipeline_equivalence.py` (per-cell Crate diag / Cpsth diag / 1-α
   r ≥ 0.99, |Δ pop-median 1-α| ≤ 0.002); corrections at population scale via
   `fig_pipeline_corrections.py`; runtime via `fig_pipeline_speed.py`.

5. **Production pipeline change for Ext-2 (still gated).** Add target-
   distribution weights to `estimate_rate_covariance` /
   `bagged_split_half_psth_covariance` / the Ctotal computation in
   `VisionCore/covariance.py`. Default `target='naive'` so current numbers
   stand. Use the §4.4 pooled-per-bin KDE construction for the multi-bin
   trajectory case. Then the expensive GPU `fig2_decomposition` cache regen.
   The estimator change is validated end-to-end in `note_pipeline.md` §7; only the GPU swap
   remains.

## Build / test commands

Run from this folder. Workspace `.venv` is at the v1-fovea repo root;
`uv run` resolves it automatically.

    # Generator self-check
    uv run python synthetic.py

    # Tests (23 tests, ~7 minutes -- random field Cholesky per time bin)
    uv run --with pytest pytest test_estimators.py -q

    # Main writeup figures
    uv run python fig_model.py
    uv run python fig_mechanism.py
    uv run python fig_distribution_truth.py
    uv run python fig_sanity_check.py
    uv run python compute_weighting_data.py     # Ext-1 real-data driver (cache-only, all sessions)
    uv run python fig_weighting_bias.py         # Ext-1 cross-cell weighting bias (Fig. 1)
    uv run python fig_recovery.py               # Fig. 3 matched-estimator recovery (§4.2)
    uv run python fig_trajectory.py             # §4.4 multi-bin extension (Fig. 4)
    uv run python fig_consistency.py            # parallel sweep, cached

    # ANOVA side note figures (note_anova.md)
    uv run python fig_anova.py                  # Fig. 1 of note_anova.md
    uv run python fig_panel_d_anova.py          # cell-side matching, ANOVA on twin
    uv run python fig_panel_d_closepair.py      # Fig. 2 of note_anova.md

    # Close-pair-density side note (note_closepair_density.md)
    uv run python compute_closepair_density.py --recompute   # cache-only, all sessions
    uv run python fig_closepair_density.py

    # Writeup (number-eqs.lua re-adds equation numbers; pandoc --mathml
    # drops amsmath \tag{})
    pandoc writeup.md      -s --mathml --self-contained --lua-filter=number-eqs.lua -o writeup.html
    pandoc note_pipeline.md -s --mathml --self-contained --lua-filter=number-eqs.lua -o note_pipeline.html
    pandoc note_anova.md    -s --mathml --self-contained --lua-filter=number-eqs.lua -o note_anova.html
    pandoc note_closepair_density.md -s --mathml --self-contained --lua-filter=number-eqs.lua -o note_closepair_density.html

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
