---
title: "Restricting the analyzed fixation range (eye-position masking)"
subtitle: "Are 1−α, the Fano correction, and the residual corrected noise correlation stable as the fixation window tightens?"
---

# Summary

This note tests how the headline LOTC-decomposition results depend on **how wide
a fixation window we admit**. For each session we freeze the fixation center at
the geometric median of the baseline-valid eye positions and re-run the
*unchanged* methods pipeline (`target='full'`) at progressively tighter fixation
radii $r\in\{1.0,0.75,0.5\}$ deg, against a no-extra-mask **baseline** column. The
estimator is best-conditioned where close pairs are dense (near the center of
fixation); the periphery is where the Direction-1 $1/\hat p$ weights blow up and
the eye tracker is least reliable. If the residual corrected noise correlation
(`z_c`), the FEM fraction $1-\alpha$, and the Fano correction **drift** as we
tighten, the periphery is driving the effect; if they **hold**, the effect is
robust.

**Bottom line.** The residual corrected noise correlation is **stable** under
tightening: at every counting window the baseline and $r=0.5$ clustered-bootstrap
CIs for `z_c` overlap and the point estimate moves by $\le 0.007$ (Fisher-$z$).
What *does* change is **estimator variance, not the central value** — close pairs
attrit far faster than samples (up to **42%** of close pairs gone at 50 ms,
$r=0.5$, vs **16%** of samples), so the CIs and the panel-3D shuffle band widen
and significance erodes at the tightest radius and longest window. The FEM
fraction $1-\alpha$ falls modestly as the truncated viewing distribution removes
the large-eccentricity excursions that drive FEM (e.g. $0.717\to0.671$ mean at
25 ms), and the corrected Fano slope rises slightly (less correction). None of
these is a regime change. **Restricting the analyzed fixation range does not
overturn any headline result; it only costs precision.**

This is a read-only vetting pass. Production (`VisionCore/covariance.py`,
`ryan/fig2`, `ryan/fig3`, `ryan/fig4`) and the methods estimator (`pipeline.py`,
`estimators.py`, `metrics.py`) were **not modified** — the masking is applied to
the aligned input and everything downstream is reused as-is. All new code lives
under `methods_eyepos_matching/eyepos_masking/`.

---

# 1 — Masking specification and the linchpin check

## 1.1 The mask

For each of the 25 sessions (Allen ×11, Logan ×14) in
`cache/aligned_sessions.pkl`:

1. **Center = geometric median** of all baseline-valid per-(trial, bin) eye
   positions, `center = _geometric_median(eyepos[valid_mask])` (one 2-vector per
   session, **held fixed across all radii**).
2. **Radii** $r\in\{1.0,0.75,0.5\}$ deg, plus a no-extra-mask **baseline**.
3. **Whole-trajectory rule:** a decomposition sample is dropped if *any* bin of
   its `t_hist + t_count` trajectory leaves the radius-$r$ disk. Implemented by
   tightening the per-bin validity mask and re-running the pipeline:

   ```python
   inside       = np.linalg.norm(aligned["eyepos"] - center, axis=-1) <= r
   valid_mask_r = aligned["valid_mask"] & inside
   ```
4. **Neuron set held fixed** — baseline `rate_hz`, `psth_r2`, `contam_rate`,
   `neuron_mask` are unchanged across radii; only the eye-position sample mask
   (and the covariance it produces) varies.

The baseline-valid mask is a clean subset of the finite-eye-position bins (0
exceptions across all sessions), so the geometric median and the `inside` mask
are well defined and NaN bins (already invalid) map to `inside=False`.

## 1.2 Linchpin: mask-tightening ⇔ "any part of the trajectory outside ⇒ drop"

`extract_valid_segments` forms windows only inside contiguous valid runs and
`_extract_windows_numpy` emits a window only if its full trajectory lies in one
segment. So marking out-of-disk bins invalid must exclude exactly those windows
whose trajectory touches an out-of-disk bin. `verify_equivalence.py` confirms
this directly:

- **(A) Soundness.** Across **300** (session × window × radius) cells, *every*
  retained window's trajectory lies entirely inside its disk
  (`max ||traj − center|| ≤ r`). PASS.
- **(B) Fragmentation side-effect.** Tightening can also split a long valid run
  into pieces shorter than `min_seg_len=36`, dropping windows that are *fully
  inside* the disk. Summed over sessions × windows, this costs **4017 / 5964 /
  14571** windows at $r=1.0/0.75/0.5$ — a real, reported consequence of working
  inside contiguous segments, separate from the disk rule itself.
- **(C) Monotone nesting.** Inside-disk bit-masks nest
  $r{=}0.5\subset0.75\subset1.0$ and retained sample/close-pair counts are
  non-increasing under tightening, on every session. PASS.

At the standard config `t_hist_ms=10` the history is **1 bin**, so the
whole-trajectory rule is close to a per-count-bin mask; we implement the general
rule (all `t_hist + t_count` bins inside) and quantify the difference in §1.4.

## 1.3 Fixation geometry

| quantity | median | range |
|---|---|---|
| geo-median offset from target $(0,0)$ | 0.045 deg | 0.017 – 0.167 |
| fixation spread $\sigma_e$ (RMS about center) | 0.323 deg | 0.228 – 0.385 |

The geo-median offsets are small (max 0.167 deg), so **$r=1.0$ around the
geo-median is nearly identical to the production `FIXATION_RADIUS = 1.0` around
the target** — it drops only ~0.1% of samples (next section), which is why the
baseline and $r=1.0$ columns are visually indistinguishable everywhere. Allen
sessions are slightly more spread ($\sigma_e\approx0.30$–$0.39$) than Logan
($\approx0.23$–$0.38$). Per-session values: `masking_stats.py`.

## 1.4 Attrition — population, per window

`samples` = decomposition windows; `clpairs` = close pairs (the quantity that
actually drives the close-pair $C_\text{rate}$); `trajExtra%` = share of
count-bin-inside windows additionally dropped because a *non-count history* bin
left the disk (the cost of the whole-trajectory rule over a per-count-bin mask).

| win (ms) | radius | samples | % drop | close pairs | % drop | trajExtra % |
|---|---|---|---|---|---|---|
| 8.3 | r=1.0 | 143601 | 0.1 | 71291 | 0.0 | 0.02 |
| 8.3 | r=0.75 | 138867 | 3.4 | 70825 | 0.7 | 0.28 |
| 8.3 | r=0.5 | 122055 | 15.1 | 67155 | 5.8 | 0.94 |
| 16.7 | r=0.75 | 68442 | 3.5 | 26390 | 10.7 | 0.43 |
| 16.7 | r=0.5 | 60013 | 15.4 | 21563 | 27.0 | 1.40 |
| 25.0 | r=0.75 | 44965 | 3.5 | 15104 | 13.9 | 0.59 |
| 25.0 | r=0.5 | 39330 | 15.6 | 10916 | 37.8 | 1.95 |
| 50.0 | r=0.75 | 21500 | 3.8 | 5412 | 17.1 | 1.13 |
| 50.0 | r=0.5 | 18678 | 16.4 | 3787 | 42.0 | 3.70 |

Three facts set up everything downstream:

- **Sample attrition is window-flat** (~0.1 / 3.5 / 15–16% at $r=1.0/0.75/0.5$):
  it is set by how much of the fixation mass sits inside the disk, the same for
  any counting window.
- **Close-pair attrition is far steeper and grows with window** (5.8% → **42%**
  at $r=0.5$ from 8.3 → 50 ms). Close pairs require *both* trajectories to
  survive, and the RMS close-pair criterion gets harder to satisfy as the
  trajectory lengthens; tightening the disk removes the eccentric tails where
  many long-window close pairs lived. **This is the dominant effect of masking**
  — it inflates estimator variance even where the sample count is barely touched.
- **The whole-trajectory rule is cheap but non-trivial at long windows**
  (`trajExtra` ≤ 0.1% at 8.3 ms, up to 3.7% at 50 ms / $r=0.5$): with only one
  history bin almost all the drop is from the count window itself, as expected.

**Session survival.** No session falls below `valid.sum() < 3` or to zero close
pairs at any radius; all 25 sessions retain all four windows, so the population
panels keep $n=25$ throughout. Segment counts are roughly stable (net $-0.1$% at
$r=1.0$ — tightening can *split* one run into two, momentarily raising the count
— to $-3.7$% at $r=0.5$); the meaningful attrition is in samples and close pairs
above.

## 1.5 Baseline reproduces the existing pipeline

The $r=\infty$ baseline is run through the **identical** masking code path
(mask unchanged). It reproduces the existing `cache/methods_derived.pkl` to
`max |Δ| = 0.00e+00` on the Fano `slope_cor` and NC `dz_mean` across all targets
and windows — confirming the masking harness changes nothing when the mask is a
no-op, so every difference below is the disk, not the plumbing.

---

# 2 — Panels and movement

## 2.1 4-up across radii (target = full)

Each figure renders **BASELINE | r=1.0 | r=0.75 | r=0.5** by calling the *same*
production panel function four times with four per-radius data dicts (built
through the consistency adapter), so any visible difference is in the data, not
the drawing. Each radius's panel-3D shuffle band is that radius's **own** naive
eye-shuffle null (`null_from='naive'` within its bundle) — the null's width
legitimately changes as samples are removed, so it must be recomputed per radius
rather than borrowed from baseline. A `target='naive'` 4-up is emitted as a
cross-check (`figures/eyepos_masking/mask_*_naive.png`).

![Fig 2C — per-subject $1-\alpha$ histogram at 8.3 ms. The distribution shifts
gently left as $r$ tightens: a truncated viewing distribution attributes a
smaller fraction of rate variance to FEM.](../figures/eyepos_masking/mask_fig2c_full.png)

![Fig 2E — population Fano slope-through-origin vs window. Corrected slopes
(filled/solid) rise slightly as $r$ tightens (less correction); the CIs widen at
$r=0.5$, most at 50 ms where close pairs are scarcest.](../figures/eyepos_masking/mask_fig2e_full.png)

![Fig 3B — corrected vs uncorrected noise-correlation scatter at 8.3 ms, one
sub-panel per monkey. The clouds are stable across radii; only their spread
grows at $r=0.5$.](../figures/eyepos_masking/mask_fig3b_full.png)

![Fig 3C — mean Fisher-$z$ noise correlation vs window. The corrected curve
(solid) holds its level across radii while its error bars widen markedly at
$r=0.5$ — variance, not bias.](../figures/eyepos_masking/mask_fig3c_full.png)

![Fig 3D — $\Delta z$ (corrected − uncorrected) vs window against each radius's
**own** naive shuffle null. As $r$ tightens the null band balloons (fewer close
pairs) and significance erodes at the long window, even though the observed
$\Delta z$ barely moves.](../figures/eyepos_masking/mask_fig3d_full.png)

## 2.2 Movement table (target = full, clustered-bootstrap 95% CIs)

Read off the production statistics' own clustered bootstrap (`slope_cor_ci`,
`z_c_ci`, `dz_ci`, alpha `ci`) via the adapter. `z_c` = residual corrected NC;
`dz` = reduction (corrected − uncorrected).

| win (ms) | radius | $1-\alpha$ mean [CI] | $1-\alpha$ med | Fano slope_cor [CI] | NC $z_c$ [CI] | NC $\Delta z$ [CI] |
|---|---|---|---|---|---|---|
| 25.0 | BASE | 0.717 [0.706,0.728] | 0.774 | 0.856 [0.718,0.997] | 0.020 [0.006,0.034] | −0.070 [−0.083,−0.056] |
| 25.0 | r=1.0 | 0.715 [0.704,0.726] | 0.771 | 0.858 [0.719,1.018] | 0.019 [0.006,0.033] | −0.070 [−0.084,−0.057] |
| 25.0 | r=0.75 | 0.697 [0.686,0.708] | 0.750 | 0.895 [0.774,1.007] | 0.025 [0.008,0.041] | −0.061 [−0.075,−0.047] |
| 25.0 | r=0.5 | 0.671 [0.660,0.683] | 0.723 | 0.905 [0.807,0.998] | 0.013 [−0.007,0.031] | −0.064 [−0.081,−0.046] |
| 50.0 | BASE | 0.749 [0.737,0.761] | 0.817 | 1.039 [0.924,1.149] | 0.062 [0.032,0.090] | −0.064 [−0.091,−0.039] |
| 50.0 | r=0.5 | 0.707 [0.693,0.721] | 0.772 | 1.192 [1.024,1.337] | 0.061 [0.017,0.098] | −0.055 [−0.098,−0.018] |

(Full four-window × four-radius table: `movement_table.py`.) Reading the
columns:

- **$1-\alpha$ falls modestly** as $r$ tightens (25 ms: 0.717 → 0.671 mean,
  0.774 → 0.723 median at $r=0.5$). Truncating the viewing distribution removes
  the large-eccentricity excursions that produce the most FEM rate modulation,
  so a smaller fraction of across-trial rate variance is FEM-attributable. The
  shift is gradual and the CIs stay tight (samples are abundant).
- **Corrected Fano slope rises** (25 ms: 0.856 → 0.905; 50 ms: 1.039 → 1.192) —
  less correction under the tighter window — with visibly wider CIs at $r=0.5$.
  At 50 ms the corrected slope exceeds 1 already at baseline (the long-window
  over-correction regime), and tightening accentuates it.
- **$\Delta z$ shrinks slightly in magnitude** but every radius's CI overlaps
  baseline's.

## 2.3 The headline question — is the residual corrected NC stable?

| window | $z_c$ baseline | $z_c$ $r{=}0.5$ | $\Delta$ | base/$r{=}0.5$ CIs overlap | verdict |
|---|---|---|---|---|---|
| 8.3 ms | 0.011 | 0.008 | −0.003 | yes | **stable** |
| 16.7 ms | 0.017 | 0.018 | +0.002 | yes | **stable** |
| 25.0 ms | 0.020 | 0.013 | −0.007 | yes | **stable** |
| 50.0 ms | 0.062 | 0.061 | −0.001 | yes | **stable** |

At **every** window the residual corrected noise correlation is statistically
unchanged from baseline to the tightest disk: the clustered-bootstrap CIs overlap
and the point estimate moves by at most 0.007 (Fisher-$z$). The residual positive
corrected NC documented in `note_consistency.md` (the naive estimator's
$p^2$-vs-$p$ over-subtraction no longer pulling $z_c$ to zero under the matched
estimator) **is not a periphery artifact** — restricting the analyzed fixation
range to the well-conditioned center leaves it intact. What tightening *does* do
is widen the CIs (e.g. $z_c$ at 25 ms: $[0.006,0.034]$ at baseline →
$[-0.007,0.031]$ at $r=0.5$, now straddling zero) purely because close pairs have
attrited by ~38–42% at the long windows. **The effect is robust; the precision is
not.** If a central-fixation restriction is desired for other reasons, $r=0.75$
is the sweet spot — it removes the eccentric tail (samples −3.5%, close pairs
−11–17%) while keeping every $z_c$ CI tight and significant.

**Note on the truncated viewing distribution.** Tightening the window truncates
$p(e)$, so the `full`/`central` targets now match a *truncated* distribution.
This is intended — it is precisely "the effect of restricting the analyzed
fixation range" — so the $1-\alpha$ and Fano movement above should be read as a
property of the chosen analysis window, not a bug.

---

# Success criteria — status

| criterion | status |
|---|---|
| Masked caches for $r\in\{1.0,0.75,0.5\}$, geo-median center, whole-trajectory rule | done — `build_masked_caches.py`, `cache/methods_derived_r*.pkl` |
| `valid_mask`-tightening ⇔ trajectory-drop equivalence verified | done — 300 cells, soundness + nesting PASS (§1.2) |
| Per-session × radius attrition + fixation-geometry table | done — `masking_stats.py` (§1.3–1.4) |
| 4-up across-radii figures for fig2 C/E + fig3 B/C/D (full; naive cross-check) | done — `figures/eyepos_masking/` (§2.1) |
| Per-window movement table with CIs for $1-\alpha$, Fano slope, NC $z_c$/$\Delta z$ | done — `movement_table.py` (§2.2) |
| Verdict on residual corrected NC stability | done — **stable** at every window (§2.3) |
| Production + methods estimator untouched | verified — only `eyepos_masking/` added |

## Reproduce

```bash
uv run python eyepos_masking/verify_equivalence.py   # linchpin mask<->drop check
uv run python eyepos_masking/build_masked_caches.py  # per-radius derived caches
uv run python eyepos_masking/masking_stats.py        # attrition + geometry tables
uv run python eyepos_masking/make_panels.py          # 4-up figures (full + naive)
uv run python eyepos_masking/movement_table.py       # movement table + verdict
pandoc note_eyepos_masking.md -s --mathml --self-contained \
    --lua-filter=../number-eqs.lua -o note_eyepos_masking.html
```
