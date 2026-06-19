# Task brief — vet the effect of restricting the fixation range (eye-position masking)

**This file is a kickoff prompt for a fresh session. Read it, then build the
deliverable described at the bottom: a new writeup `note_eyepos_masking.md`
(plus scripts and figures) in a new `eyepos_masking/` subfolder of
`methods_eyepos_matching/`.**

Read `note_consistency.md` and `note_consistency_task.md` first — this task
reuses that session's adapter and panel infrastructure wholesale.

---

## Why we're doing this

We have just vetted the new methods pipeline against production
(`note_consistency.md`): `target='naive'` reproduces production, and the
distribution-matched `target='full'`/`'central'` are the intended replacement.
A key consequence surfaced there: under the matched estimator the **corrected
noise correlation does not go all the way to zero** (residual Fisher-$z\approx
+0.02$), because the naive close-pair estimator was over-subtracting the rate
covariance (the $p^2$-second-moment / $p$-mean inconsistency of writeup §4.1).

Before propagating the new pipeline into production we want one more robustness
check: **how do the headline results depend on how wide a fixation window we
admit?** The estimator is best-behaved where close pairs are dense and the
importance weights are well-conditioned — i.e. near the center of fixation. The
periphery is exactly where Direction-1 (`full`) $1/\hat p$ weights blow up
(writeup §4.3) and where the eye tracker is least reliable. Restricting to a
tighter fixation disk should (a) reduce estimator variance and (b) test whether
the residual corrected NC, the FEM fraction $1-\alpha$, and the Fano correction
are **stable** as we tighten the window — or whether they drift, which would
tell us the periphery is driving (or contaminating) the effect.

Concretely we want, for each session, to recenter on the fixation **geometric
median** and re-run the whole panel suite at progressively tighter fixation
radii, reporting both the scientific panels and the data attrition at each step.

## The masking specification (implement exactly)

For **each session** in `cache/aligned_sessions.pkl`:

1. **Center of fixation = geometric median.** Compute the L1 multivariate median
   of all *baseline-valid* per-(trial, time-bin) eye positions in that session:
   `center = _geometric_median(eyepos[valid_mask])` (reuse
   `estimators._geometric_median`, estimators.py:505). One 2-vector per session,
   held **fixed** across all radii.

2. **Radii:** $r \in \{1.0,\ 0.75,\ 0.5\}$ degrees, plus a **baseline**
   (no extra mask = the current pipeline) as the reference column.

3. **Masking rule — "any part of the trajectory outside the disk drops the whole
   sample."** A decomposition sample at count-bin $t$ uses a trajectory spanning
   `t_hist + t_count` contiguous bins (`pipeline._extract_windows_numpy`); the
   sample must be dropped if **any** bin of that trajectory lies outside the
   radius-$r$ disk around `center`. **Recommended clean implementation:** tighten
   the per-bin validity mask,

   ```python
   inside = np.linalg.norm(aligned["eyepos"] - center, axis=-1) <= r   # (n_trials, T)
   valid_mask_r = aligned["valid_mask"] & inside
   ```

   then run `decompose_session` on a session copy with `valid_mask = valid_mask_r`.
   Because `extract_valid_segments` only forms windows inside **contiguous valid
   runs** (`min_seg_len=36`), marking out-of-disk bins invalid automatically
   excludes every window whose trajectory touches an out-of-disk bin — which is
   exactly the "any part outside ⇒ drop" rule. (Verify this equivalence with a
   small unit check; it is the linchpin of the whole task.)

4. **Hold the neuron set fixed.** Keep the baseline per-cell inclusion
   (`rate_hz`, `psth_r2`, `contam_rate`, `neuron_mask`) from
   `aligned_sessions.pkl` **unchanged** across radii, so the only thing varying
   is the eye-position sample mask and the covariance it produces. (Recomputing
   `rate_hz`/`psth_r2` under each masked pool is a defensible alternative —
   report how much they would move, but do **not** let it change the neuron set,
   or the panels stop being comparable across radii.)

## Panels in scope (same five as the consistency note)

| panel | file | shows |
|---|---|---|
| fig2 **C** | `ryan/fig2/generate_fig2c.py` | per-subject $1-\alpha$ histogram (primary window) |
| fig2 **E** | `ryan/fig2/generate_fig2e.py` | population Fano slope vs window |
| fig3 **B** | `ryan/fig3/generate_fig3b.py` | NC scatter (corrected vs uncorrected) |
| fig3 **C** | `ryan/fig3/generate_fig3c.py` | mean Fisher-$z$ NC vs window |
| fig3 **D** | `ryan/fig3/generate_fig3d.py` | $\Delta z$ vs window vs shuffle null |

For each panel produce a **4-up across radii**: **BASELINE | r=1.0 | r=0.75 |
r=0.5**, at **`target='full'`** (the production default). Optionally also emit a
second 4-up at `target='naive'` as a cross-check. This is the same column-swap
trick as `consistency/make_panels.py` — only the data dicts change.

## Reuse the consistency infrastructure (do NOT rewrite it)

The previous session built everything you need to render panels from an
alternate pipeline; import and reuse it:

- **`consistency/adapter.py::methods_to_fig2_schema(md, target, null_from=)`** —
  maps a `derive_methods` bundle into the exact `load_fig2_data()` schema by
  running the methods `metrics[target]` through the **literal production**
  Stage-3 functions. Feed it each per-radius `derive_methods` bundle. Use
  `null_from='naive'` for panel-3D's reference band (see below).
- **`consistency/make_panels.py`** — copy its `panel_func()` import-hygiene
  helper and the gridspec/`fig.text` 4-up layout verbatim; just change `COLS`
  and `DATA` to be the radii instead of the targets.
- **Panel functions** `plot_panel_X(ax=…, data=…)` render unchanged.

Pipeline entry points (unchanged from the consistency task):

- `pipeline.decompose_session(aligned, windows_bins=(1,2,3,6), targets=('naive',
  'full','central'), t_hist_ms=10, n_shuffles=100, …)` → per-session result.
- `metrics.derive_methods(session_results, windows_ms, windows_bins, targets)` →
  the bundle the adapter consumes.
- `cache/aligned_sessions.pkl`: list of 25 sessions (Allen ×11, Logan ×14),
  each with `eyepos (n_trials,120,2)`, `valid_mask (n_trials,120) bool`, `robs`,
  `neuron_mask`, `rate_hz`, `psth_r2`, `contam_rate`. **No Luke** in the data.

## Statistics to report (Phase 1 of the note)

Per session × radius (and a population roll-up):

- **Fixation geometry:** geo-median `center` (deg), its offset from the fixation
  target $(0,0)$, and the per-session fixation spread $\sigma_e$.
- **Attrition:** baseline valid samples → retained samples at each $r$, as count
  **and fraction dropped**; same for **close pairs** (the quantity that actually
  drives the close-pair Crate) and for **valid time-bins / segments** (how many
  segments fall below `min_seg_len=36` and vanish entirely).
- **Trajectory-vs-bin drop share:** of the dropped samples, how many were lost
  because a *non-count* history bin (not the count window itself) left the disk —
  this quantifies how much the "whole-trajectory" rule costs over a naive
  per-count-bin mask.
- **Session survival:** any session that falls below the `valid.sum() < 3`
  threshold (or has zero close pairs) at a given radius must be flagged and
  dropped, with the population panels noting the changed $n$.

Note the existing `data_loading.FIXATION_RADIUS = 1.0` is applied around the
fixation **target**, not the per-session geo-median, so even $r=1.0$ around the
geo-median will drop a few samples wherever the median is offset — report that
offset so the $r=1.0$ attrition is interpretable.

## Phase 2 — panels and movement

For each panel, the 4-up across radii. Then a movement table, per window, of the
quantities we care about as $r$ tightens:

- $1-\alpha$ (mean/median), Fano `slope_cor`, NC `z_c_mean` (residual corrected
  correlation) and `dz_mean` (reduction). **The headline question:** does the
  residual `z_c` stay ~constant as the window tightens (effect is robust), shrink
  toward zero (periphery was inflating it), or grow (periphery was suppressing
  it)? Report with bootstrap CIs so "stable vs drifting" is a statistical
  statement, not eyeballing.

**Panel-D shuffle null.** The pipeline computes the eye-shuffle null for
`target='naive'` only, and only on each radius's own masked data. Use each
radius's naive null as that radius's panel-3D band (`null_from='naive'` within
that radius's bundle) — do **not** borrow the baseline null across radii, since
the null's width legitimately changes as samples are removed. Document this.

## Success criteria

- Masked caches built for $r\in\{1.0,0.75,0.5\}$ with the geo-median center and
  the whole-trajectory rule, plus the verified `valid_mask`-tightening ⇔
  trajectory-drop equivalence check.
- A per-session × radius attrition + fixation-geometry table.
- 4-up across-radii figures for fig2 C/E and fig3 B/C/D (target `full`; naive
  optional), saved under `figures/eyepos_masking/`.
- A per-window movement table for $1-\alpha$, Fano slope, NC `z_c`/`dz` with CIs,
  and a clear verdict on whether the residual corrected NC is stable under
  tightening.

## Deliverable

A new writeup **`eyepos_masking/note_eyepos_masking.md`** (build to HTML the same
way as the others: `pandoc note_eyepos_masking.md -s --mathml --self-contained
--lua-filter=../number-eqs.lua -o note_eyepos_masking.html`), plus the
cache-building / panel scripts under `eyepos_masking/` and the
`figures/eyepos_masking/` figures.

**Do not modify production** (`VisionCore/covariance.py`, `ryan/fig2`,
`ryan/fig3`, `ryan/fig4`) — read-only vetting, evidence only. Also do not modify
the methods estimator (`pipeline.py`, `estimators.py`, `metrics.py`); the masking
is applied to the aligned input and everything downstream is reused as-is.

## Gotchas / notes

- Branch before any edits (don't work on `main`); the work lives inside the
  `VisionCore` submodule. Use `uv run` for everything (`python-environment`
  skill). BLAS-pin parallel workers as `compute_methods_data.py` does.
- Re-running `decompose_session` for 25 sessions × 3 radii × 3 targets with
  shuffles is the expensive step — parallelize per (session, radius) and cache
  each radius's `derive_methods` bundle (`cache/methods_derived_r{R}.pkl`) so the
  panel/figure iteration is fast. Mirror `compute_methods_data.py`'s worker setup.
- At the standard config `t_hist_ms=10` (~1 bin) the trajectory is short, so the
  "whole-trajectory" rule is close to a per-bin mask — but implement the general
  rule (all `t_hist + t_count` bins inside) so it stays correct if `t_hist` grows.
- The geo-median must be computed on the **baseline** valid pool and frozen; do
  not recompute it per radius (that would move the center as you mask).
- Tightening the window truncates $p(e)$, so the `full`/`central` targets now
  match a *truncated* viewing distribution — this is intended (it is the whole
  point), but say so in the note so the $1-\alpha$/NC movement is read as "effect
  of restricting the analyzed fixation range," not a bug.
- Sanity check one session by hand: confirm retained-sample count equals
  `(valid_mask & inside).sum()`-derived window count, and that $r=0.5\subset
  r=0.75\subset r=1.0$ nest (monotone attrition).
