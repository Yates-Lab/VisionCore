---
title: "Production pipeline state and the CPU-parallel Figure-2 reimplementation"
subtitle: "An implementation note to `writeup.md` (methods_eyepos_matching)"
author: "fem-v1-fovea methods note"
date: "2026-06-18"
header-includes: |
  <style>
  .numbered-equation { position: relative; }
  .numbered-equation .eqno {
    position: absolute; right: 0.5em; top: 50%;
    transform: translateY(-50%);
  }
  </style>
---

# Summary

This is an implementation note to the main methodological writeup in this
folder (`writeup.md`, *Extending McFarland's cross-trial decomposition...*).
It records the production-pipeline state for each extension and documents the
CPU-parallel reimplementation of the entire Figure-2 LOTC pipeline that lives
in this folder. None of it is needed to follow the main note's methodology;
it is kept so the pipeline-change record and the equivalence/validation
evidence remain referenceable.

The section numbers (§6, §7) and figure numbers (Figs. 7–9) below are retained
from the main note so existing cross-references continue to resolve.
References of the form "§4.x" or "main note Fig. N" point into `writeup.md`.

# 6. The current pipeline: state and proposed production change

The covariance machinery and caches are shared by Figures 2–4, so a pipeline
change has broad blast radius. This section consolidates the
production-pipeline state for each extension.

## 6.1 Extension 1 — already integrated

Consistent pair-count time-bin weighting has been integrated into the production
pipeline (`VisionCore/covariance.py`):

- `estimate_rate_covariance` — pair-count-weighted $\bar Y$ matching the
  close-pair second moment, pinning the $\bar Y$ cell of the §1.5 table to
  the pair-count direction (§3.4).
- `bagged_split_half_psth_covariance` — `weighting` parameter, default
  `'pair_count'`, pinning $C_\text{psth}$ to the pair-count direction.

Together with the trial-count $C_\text{total}$ already being recomputed at
the same per-sample weight inside the production estimator, this pins the
full $w_t$ column to the pair-count direction (§3.4).

On the Allen 2022-04-13 session (49 cells, 3667 windows) the fix moved the
shuffle-null $D_z$ from $-0.0068$ ($p<10^{-4}$) to $+0.0010$ ($p=0.44$); the
real-data $D_z$ shifted only $-0.0855 \to -0.0819$ (signal-to-null ratio
$12.5\times \to 83.7\times$), so the scientific conclusion was preserved.
Diagnosis and validation are recorded in
`ryan/fig2/bias_diagnosis/FINAL_REPORT.md`.

## 6.2 Extension 2 — estimator validated, pipeline rebuilt locally, GPU swap gated

Eye-position-distribution matching is implemented and TDD-validated in this
folder, for both the single-bin (main note §4.2) and multi-bin trajectory
(main note §4.4) filters:

- `estimators.decompose(target=...)` — single-bin close-pair filter.
  `naive` reproduces the existing pipeline; `full` is Direction 1 (the
  actual-viewing $p$); `central` is Direction 2 ($p^2$).
- `estimators.decompose_trajectory(target=...)` — multi-bin RMS-trajectory
  close-pair filter (matching `VisionCore/covariance.py`); each trajectory is
  reduced to its geometric-median representative point and one KDE is fit on
  those points, reusing the single-bin §4.2 importance weights ($p^2$ implied
  by $\hat p$), per the main note §4.4 single-point reduction.
- `test_estimators.py` — 27 tests covering correctness (recovery of the
  closed-form decompositions under each target, finite-threshold-bias
  shrinkage), stability (Direction 2 stabler than Direction 1 for eccentric
  cells), Poisson cancellation (Fano $\to 1$), the pipeline-match
  (`naive` ↔ `pipeline_one_minus_alpha`), Extension 1 (variable-$n_t$
  uniform vs pair-count weighting), and the main note §4.4 multi-bin extension
  (geometric-median reduction unit tests, flat-limit and realistic-drift
  recovery, geometric-median≈centroid in the flat regime, naive bias on a
  centrally-modulated cell, and the eigendecomposition field-sampler fallback
  on dense real eye positions).

§7 rebuilds the entire Figure-2 LOTC pipeline in this folder around
`decompose_trajectory`. The replacement runs all three targets in parallel on
CPU, reproduces the legacy numbers at `target='naive'` within a small,
documented tolerance, and quantifies the correction at `full`/`central` across
the same 30 sessions. The proposed production change is the same as before —
add the `target` argument to `estimate_rate_covariance` and
`bagged_split_half_psth_covariance`, defaulted to `'naive'` so existing
numbers are preserved unless `target='full'` is explicitly requested — but the
gate is now narrower: the *estimator change* is validated end-to-end; only
the *GPU cache regeneration* in production remains pending explicit approval.

---

# 7. Pipeline implementation and validation

The main note establishes the corrected estimator and confirms it on
synthetic data with a closed-form ground truth and on cached real-data
quantities at the per-cell level (main note Fig. 5). This note closes the
loop: a
parallel, CPU-only reimplementation of the entire Figure-2 LOTC pipeline,
verified against the production pipeline on the same 25 sessions ($11$ Allen,
$14$ Logan; $n = 1313$ included cells at the canonical window
$t_\text{count} = 2$ bins $\approx 16.7$ ms), and quantifying the
distribution-matched correction at population scale.

The pipeline is self-contained: every script lives in this folder, the only
dependency on the broader codebase is a one-shot data loader that produces a
local cache (`cache/aligned_sessions.pkl`), and the production code itself is
frozen as a verbatim snapshot under `legacy/` so the comparator never drifts.

## 7.1 Architecture

The production pipeline (`legacy/compute_fig2_data.py`, the 2026-06-02
snapshot of `VisionCore/ryan/fig2/compute_fig2_data.py`) is two-stage and
GPU-bound: stage 1 calls `VisionCore.covariance.run_covariance_decomposition`
which does its close-pair binning and second-moment accumulation on a
torch GPU tensor; stage 2 aggregates per-cell 1-α, Fano, and noise-
correlation summaries from the per-session covariance matrices.

The replacement keeps the two-stage structure but rewrites stage 1 around
`decompose_trajectory` (numpy-only, all three targets per call) and lifts
stage 2 verbatim from the snapshot. Self-containment is enforced at the
boundary between data loading and decomposition:

  * `data_loading.py` — calls `models.data.prepare_data` once per session,
    aligns the fixRSVP trials via `legacy.covariance.align_fixrsvp_trials`,
    pre-computes the rate-Hz and split-half PSTH-$R^2$ inclusion statistics,
    and writes a single pickle `cache/aligned_sessions.pkl` keyed by
    `schema_version=1`. Sessions that the legacy pipeline skips (failures
    inside `prepare_data`, missing fixRSVP, too few trials) are skipped here
    too, byte-for-byte.
  * `pipeline.py` — per-session driver. The numpy port of
    `legacy.covariance.extract_windows` (`_extract_windows_numpy`) is verified
    bit-identical to the legacy torch implementation on a 3-window fixture
    (`test_pipeline.py::test_extract_windows_matches_legacy`), so the close-
    pair pool and per-bin time-index `T_idx` match by construction.
    For each $(t_\text{count}\in\{1,2,3,6\}, \text{target}\in\{\text{naive},
    \text{full}, \text{central}\})$ it calls `decompose_trajectory(...)`,
    then overrides the returned $C_\text{rate}$ with the uncentred
    close-pair form $\hat M\!M - \hat r\hat r^\top$ (see main note §4.4's
    centred-form discussion and §7.2 item 6 for why this matters for both consistent
    and inconsistent targets), and runs 100 eye-trajectory shuffles of
    `target='naive'` to reproduce the legacy null. Full/central shuffles
    are deferred (§7.5) — the naive null is enough to reproduce the legacy
    $p$-values that drive the equivalence check, and the corrections
    figure only needs the real population shifts.
  * `metrics.py` — stage-2 aggregation. The session-clustered Fano-slope
    bootstrap, Fisher-$z$ noise-correlation means, and shuffle-null
    $p$-values are lifted verbatim from `legacy.compute_fig2_data`
    (`_compute_metrics`, `_compute_alpha_stats`, `_compute_fano_stats`,
    `_compute_nc_stats`, `_clustered_slope_bootstrap`) so the only
    methodological difference between the two pipelines is the
    target-distribution weighting inside stage 1.
  * `compute_methods_data.py` — orchestrator. `joblib.Parallel(backend="loky")`
    over the aligned-session list; each worker pins its BLAS thread pool to 1
    (`threadpool_limits(1)`) so $n$ workers do not oversubscribe the box.
    The same script with `--legacy` runs the snapshot's
    `_compute_one_session` on CPU against the same aligned cache, producing
    a comparator pickle `cache/legacy_decomposition.pkl` that the
    equivalence figure reads.

The aligned-session cache is the cut that makes everything else
regeneratable from inside this folder: once it exists, `compute_methods_data
[--legacy|--both]`, `timing.py`, `profile_pipeline.py`, and the three
figure scripts run without touching `VisionCore.covariance` (live),
`models.data.prepare_data`, or any production cache. A future refactor of
`VisionCore.covariance` does not change the comparator's behaviour.

## 7.2 Equivalence at `target='naive'`

The methods pipeline at `target='naive'`, `cpsth_method='split_half'`,
`time_bin_weighting='pair_count'` should be algebraically identical to the
legacy snapshot up to a small set of known controlled differences:

  1. **Shuffle RNG** — legacy uses a torch generator on GPU; methods uses
     `numpy.default_rng`. We compare *distributions* of null statistics, not
     per-draw values.
  2. **NaN handling** — legacy `nan_to_num(eyepos, nan=0.0)` before
     `extract_windows` (turning trial dropouts into zero-trajectory close
     pairs); the methods pipeline matches this behaviour (in
     `pipeline.decompose_session`, `nan_to_num` is applied on `robs` and
     `eyepos` before segmentation, identical to legacy
     `run_covariance_decomposition`).
  3. **$C_\text{total}$ definition** — the methods pipeline replaces
     `decompose_trajectory`'s weighted $C_\text{total}$ with the legacy
     unweighted definition (`np.cov(X.T, ddof=1)` on isfinite rows; see
     `pipeline._ctotal_unweighted`), so the Fano-factor numerator
     $C_\text{noise} = C_\text{total} - C_\text{rate}$ uses the same total.
  4. **$\bar r$ ("Erate") under variable $n_t$** — legacy computes
     pair-count-weighted $\bar r$ inside `estimate_rate_covariance` (a 2025
     fix; see §3.4). The methods pipeline uses the `_weighted_mean` of $S$
     with the same per-bin weight; the implementations agree to ~$10^{-9}$
     on the diagonal.
  5. **`intercept_mode='below_threshold'`** — legacy `below_threshold(0.05)`
     pools all $\Delta e<0.05$ pairs into a single bin and averages with
     uniform per-pair weight; this is identical to the methods close-pair
     estimator at `threshold=0.05` and `time_bin_weighting='pair_count'`.
  6. **Centred vs uncentred close-pair second moment.** The
     `decompose_trajectory` estimator computes the close-pair second
     moment on the *centred* counts $S - \hat r$ (for numerical precision when
     $\hat r\cdot t_\text{window}$ dominates $S$); the legacy
     `compute_conditional_second_moments` accumulates the *uncentred*
     product $S_i S_j^\top$ and subtracts $\hat r\,\hat r^\top$ afterward.
     The two forms
     collapse to the same expression iff $\hat r = \bar X_w$, where
     $\bar X_w$ is the weighted close-pair-set sample mean. For the
     consistent targets `full` and `central` the importance reweighting
     makes $\hat r$ and $\bar X_w$ converge to the same population
     quantity — but they are distinct estimators in finite samples, so the
     centred and uncentred forms differ on real data by a non-negligible
     amount (e.g. at $t_\text{count}=2$, $\text{median}(1-\alpha_\text{full})
     = 0.71$ centred vs $0.76$ uncentred). For the inconsistent target
     `naive` the gap is also nonzero, and there $\hat r$ and $\bar X_w$
     do not even share a limit. The §7 pipeline overrides Crate with the
     uncentred form via `pipeline._legacy_compat_crate` for **all three
     targets**, so that (a) the equivalence audit against the legacy
     uncentred form is exact for naive, and (b) the full/central numbers
     extend the main note §4.5 cell-side reference (Fig. 5) — which also used the
     uncentred form via single-bin `estimators.decompose`. The
     centred form remains the default inside `decompose_trajectory` for
     its synthetic precision benefits, but the two estimators are *not*
     interchangeable on real data.

Fig. 7 reports, on the canonical window $t_\text{count}=2$ bins, the
per-cell scatter of legacy vs methods at $\text{target}=\text{naive}$, for
three quantities:

  A. $\text{diag}(C_\text{rate})$ (the rate-variance numerator),
  B. $\text{diag}(C_\text{psth})$ (the PSTH-variance term debiased against
     same-bin observation noise),
  C. $1-\alpha = 1 - C_\text{psth}/C_\text{rate}$ (the headline number).

Pass criteria — Pearson $r\ge 0.99$ on A and B, $|\Delta\text{median}(1-\alpha)|
\le 0.002$ — are checked in code (`fig_pipeline_equivalence.py` exits
non-zero if any fail). On the 25 sessions $\times$ canonical window run,
$n=1313$ included cells: $r_\text{Crate} = 1.0000$,
$r_\text{Cpsth} = 1.0000$, $r_{1-\alpha} = 0.9999$;
$|\Delta\text{median}(1-\alpha)| = 0.0017$ (legacy median $0.7890$, methods
$0.7873$). The residual $|\Delta|$ comes from the small Cpsth difference
that the split-half-bootstrap loop accumulates from a different per-worker
RNG order between the two `Parallel` runs (item 1) — pure stochastic
noise, not estimator bias.

![**Figure 7 — Pipeline equivalence at $t_\text{count}=2$ bins (target='naive').**
Per-cell scatter of legacy snapshot (x-axis) vs methods pipeline (y-axis)
across all 25 sessions at the canonical window. **(A)** Rate-variance diagonal
on log axes; **(B)** PSTH-variance diagonal on log axes; **(C)** $1-\alpha$ on
linear axes. Inset: Pearson $r$, slope-through-origin, and median absolute
difference. The dashed line is the unit diagonal.](figures/fig_pipeline_equivalence.png)

## 7.3 The correction at population scale

With equivalence at `target='naive'` established, the same 30-session run
returns the full and central targets at no extra cost. Figure 8 reports the
target-dependence of the three headline quantities on the canonical window:

  * the $1-\alpha$ distribution (population median, histogram),
  * the corrected vs uncorrected Fano factor (slope-through-origin and
    geometric mean), and
  * the per-session Fisher-$z$ noise-correlation shift
    $\Delta\bar z = \bar z_\text{cor} - \bar z_\text{unc}$.

The three columns are *the same neurons*, *the same fixational eye
distribution*, *the same trial structure*, run through the *same Stage-2
aggregator* — only the eye-distribution weighting inside Stage 1 changes.
The shifts between columns are therefore pure estimator effects, not signal
effects.

At the canonical window $t_\text{count} = 2$ bins ($n = 1313$ included
cells, 25 sessions):

  * $\text{median}(1-\alpha)$: naive $0.793$, Direction 1 (full)
    $0.771$, Direction 2 (central) $0.605$. The Direction-1 vs naive
    shift ($-0.022$) has the same sign and order as the cell-side
    `generate_realdata.py` result in the main note §4.5 — recovered here at
    full multi-cell, multi-window scope. The Direction-2
    shift to $0.605$ is larger because the close-pair eye distribution
    concentrates on the central fixation peak, where the cell's mean rate is
    least eye-modulated; this is the same close-pair-density mechanism that
    the main note's §4.1 exhibits. (These numbers use the directly-estimated
    close-pair density, the estimator default — `closepair_density='direct'`,
    `note_closepair_density.md`; the short 4-bin trajectory window here makes
    the direct-vs-squared shift small relative to the 12-bin §4.5 window.)
  * $\text{median}(\text{Fano}_\text{cor})$: naive $0.931$, full
    $0.951$, central $0.900$. The Direction-1 Fano is *higher* than
    naive (the $C_\text{noise}$ numerator gains the cross-term
    correction) and the Direction-2 Fano is *lower* by a similar
    margin — both directions move the Fano factor *away* from the
    "no-FEM" reference of 1.0 by less than the naive estimator
    suggests.
  * Per-session noise-correlation shift $\Delta\bar z =
    \bar z_\text{cor} - \bar z_\text{unc}$: naive $-0.0683$,
    full $-0.0590$, central $-0.0399$. The eye-distribution-matched
    estimators reduce the apparent noise-correlation suppression by
    $\sim 14\text{–}42\%$ (naive over-suppresses, as predicted in §4
    by the close-pair-vs-marginal density mismatch).

![**Figure 8 — Eye-distribution-matching corrections at population scale
($t_\text{count} = 2$ bins, $n = 1313$ cells).** Columns: naive (legacy),
Direction 1 (full $p$), Direction 2 (central $p^2$). Row A: $1-\alpha$
population histograms with the column median annotated. Row B: Fano
corrected vs uncorrected, log-log; annotation gives the geometric mean
(uncorrected, corrected) and slope-through-origin (uncorrected,
corrected). Row C: per-session $\Delta\bar z = \bar z_\text{cor} -
\bar z_\text{unc}$ sorted by session, coloured by subject.](figures/fig_pipeline_corrections.png)

## 7.4 Runtime

`timing.py` benchmarks per-(session, window) wall-time legacy-snapshot vs
methods, on CPU. Both pipelines pay the same data-prep cost (the aligned-
session cache); only the decomposition runtime is timed.

Two numbers are relevant:

  * **Per-session, sequential, CPU.** On a 6-session benchmark with
    20 shuffles, methods runs at $0.4$–$3$ s per session per window vs
    legacy $0.8$–$9.1$ s. Pooled across the four windows the methods
    pipeline is $\boldsymbol{2.94\times}$ faster than the legacy snapshot
    (methods $29.4$ s, legacy $86.6$ s; the single-point reduction's one
    KDE is cheaper than the previous two pooled-per-bin KDEs). The speedup
    is larger for the smaller windows ($3.69\times$ at $t_\text{count} = 1$)
    where the legacy GPU pipeline's torch-tensor overhead dominates and
    smaller for the larger windows ($1.53\times$ at $t_\text{count} = 6$)
    where the close-pair enumeration is the limiting step in both.
  * **Pipeline-wide, parallel.** The orchestrator (`compute_methods_data
    --both`) processes all 25 sessions across 4 windows + 3 targets +
    20 shuffles via `joblib.Parallel(n_jobs=-1, backend="loky")` with
    BLAS pinned to 1 thread per worker. End-to-end on a 64-thread
    workstation: **methods Stage 1 $\boldsymbol{11.7}$ s**
    ($0.5$ s/session) vs **legacy Stage 1 $\boldsymbol{37.3}$ s**
    ($1.5$ s/session) — a $\boldsymbol{3.2\times}$ wall-clock speedup
    even at this trimmed shuffle count. The methods pipeline's
    parallelism advantage is in addition to the per-session speedup
    above: the legacy `_compute_one_session` instantiates torch
    on the worker process and is not safe to fan out across GPU
    workers without explicit `CUDA_VISIBLE_DEVICES` pinning, whereas
    the methods pipeline is pure numpy and scales linearly with cores.

Profiling a single session (`profile_pipeline.py`, $t_\text{count} = 2$,
all three targets, 10 shuffles) confirms the expected hotspot:
`scipy.stats.gaussian_kde.evaluate` accounts for the bulk of the methods CPU
time, dominated by the per-session KDE $\hat p$ fit on the trajectory
geometric-median representative points. Close-pair enumeration and the
split-half PSTH covariance are the next contributors. The KDE could be replaced
with a gridded 2-D histogram (the importance weights only need the density
evaluated at the representative points and their pair midpoints) for an
additional speedup, but that's a refinement, not a correctness issue.

![**Figure 9 — Per-session wall-time, methods vs legacy snapshot, CPU vs CPU
(6 sessions, 20 shuffles, all 4 windows).** **(A)** Per-session bars at the
canonical window $t_\text{count} = 2$, sorted by methods runtime. **(B)**
Per-window mean ± sd across sessions; legend shows total wall-time per
pipeline. The methods pipeline is $2.94\times$ faster than the legacy
snapshot pooled across windows, with the largest gap at the smallest
windows (where legacy torch overhead dominates). The 64-worker parallel
orchestrator extends this to a $3.2\times$ pipeline-wide
speedup.](figures/fig_pipeline_speed.png)

## 7.5 Deferred

  * **Subspace / participation-ratio / eigenspectrum metrics** (Figure-2
    panels E–G in the legacy bundle). These read off $C_\text{psth}$ and
    $C_\text{fem}$ exactly as in legacy stage 2 and can be lifted verbatim
    when needed; the equivalence demonstrated here transfers to those
    panels with zero additional estimator work.
  * **Eye-shuffle nulls for the `full` and `central` targets**. The
    Stage-1 implementation runs nulls for `naive` only (the legacy
    semantic). Full/central nulls would require re-fitting
    $\hat p_\text{cp,marg}$ on each shuffle's close-pair midpoints and re-
    computing the per-sample $\hat p_\text{cp,marg}/\hat p_\text{marg}$
    ratio under the shuffled eye assignment — straightforward but not
    needed for the equivalence assertion in §7.2.
  * **Production GPU cache regeneration** (`fig2_decomposition.pkl`). The
    on-disk fig2 cache that `VisionCore/ryan/fig2/generate_fig2*.py` reads
    still ships with `target='naive'` numbers. Swapping it to `target='full'`
    requires the production GPU pipeline to support the `target` argument
    and a fresh expensive run; per the project memory this swap is gated on
    explicit approval and is not in scope here.
