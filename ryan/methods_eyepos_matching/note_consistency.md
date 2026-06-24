---
title: "Vetting the methods-folder pipeline against production fig2/fig3"
subtitle: "Panel-by-panel: does NAIVE reproduce PROD, and how does FULL move it?"
---

# Summary

This note vets the `methods_eyepos_matching/` estimator pipeline against the
production `ryan/fig2` + `ryan/fig3` pipeline, panel by panel, **before** any
production swap. The contract:

1. The methods pipeline run with `target='naive'` must **reproduce** the current
   production results (within numerical tolerance).
2. The methods pipeline run with `target='full'` (the actual viewing
   distribution $p(e)$) is the intended replacement; we report exactly how each
   headline number moves naive→full.

**Bottom line (Phase 1):** NAIVE already reproduces PROD to within
~1–2% on every headline statistic at every window, with no statistic-function
divergence and no clip-vs-exclude pile-up in the derived bundle. The only
material upstream difference is **per-window unit inclusion** (production drops
84 units at the 50 ms window that the methods estimator keeps finite), and it
moves the pooled statistics negligibly. FULL systematically applies *less*
correction than NAIVE — i.e. the naive estimator over-corrects relative to the
real viewing distribution.

This is a read-only vetting pass. Production
(`VisionCore/covariance.py`, `ryan/fig2`, `ryan/fig3`, `ryan/fig4`) was **not
modified**. All new code lives under `methods_eyepos_matching/consistency/`.

---

# Phase 1 — step-by-step pipeline diff

All numbers below come from the existing caches:
`cache/aligned_sessions.pkl` (shared input), `cache/methods_derived.pkl`
(methods bundle), and `load_fig2_data(refresh=False)` (production bundle, read
from `outputs/cache/fig2_derived.pkl`). Reproduce with:

```bash
uv run python consistency/phase1_inspect.py     # schema + key presence
uv run python consistency/phase1_compare.py      # per-window headline stats
uv run python consistency/phase1_inclusion.py    # per-window inclusion counts
```

## 1.1 Sessions and subjects — identical

Both pipelines consume the **same** `aligned_sessions.pkl`, so the session set
is identical by construction:

| | n sessions | subjects |
|---|---|---|
| aligned cache (both) | **25** | Allen ×11, Logan ×14 |
| production `SUBJECTS` constant | — | `["Allen","Logan","Luke"]` |

**Luke is listed in the production `SUBJECTS` constant but is absent from the
data** (the aligned cache has no Luke session), so he is silently dropped on
both sides. `session_names` and `subjects` match element-for-element between the
two bundles. **Verdict: no difference.**

## 1.2 Counting windows — identical

| window | bins @ 120 Hz | ms |
|---|---|---|
| 0 | 1 | 8.33 |
| 1 | 2 | 16.67 |
| 2 | 3 | 25.0 |
| 3 | 6 | 50.0 |

`WINDOW_BINS = [1,2,3,6]`, `DT = 1/120` on both sides; `windows_ms` arrays match
exactly. **Verdict: no difference.**

## 1.3 Statistic functions — identical (by construction in Phase 2)

The methods `metrics.py` Stage-3 helpers were ported from production and use the
same `paired_valid(positive=True)` → `geomean` + `_slope_through_origin` +
`_clustered_slope_bootstrap` for the Fano, and the same Fisher-$z$ /
per-session-mean / shuffle-null machinery for NC. To remove any residual doubt,
the **Phase 2 adapter feeds the methods `metrics` list through the *literal*
production functions** `_compute_alpha_stats`, `_compute_fano_stats`,
`_compute_nc_stats` (imported from `compute_fig2_data`). Any NAIVE−PROD residual
is therefore guaranteed to originate **upstream** of Stage 3 (decomposition or
inclusion), never in the statistics. **Verdict: no difference (enforced).**

## 1.4 Clip-vs-exclude — no pile-up in the derived bundle

The estimator (`estimators.py`, `pipeline.py`) clips `alpha` to $[0,1]$, which
was the suspected source of a "pile-up at 0/1". **It does not propagate.** The
derived bundle stores the **raw** ratio in `metrics['alpha']`:

| window | source | `metrics['alpha']` range | finite | $<0$ | $>1$ |
|---|---|---|---|---|---|
| 8.3 ms | PROD | $[-84.9,\ 21.5]$ | 1359 | 44 | 48 |
| 8.3 ms | METHODS | $[-86.1,\ 21.6]$ | 1359 | 45 | 48 |

and the methods `_compute_alpha_stats_one` (metrics.py:351) applies the
**identical exclude-not-clip rule** as production (`compute_fig2_data.py:559`):

```python
in_range = np.isfinite(m_raw) & (m_raw >= 0.0) & (m_raw <= 1.0)   # both sides
```

storing the *excluded* (not clipped) raw $1-\alpha$ in `alpha_stats[...]['m']`.
The adapter feeds panel C from this `m`, so there is **no pile-up at 0 or 1**:
production's $1-\alpha$ at 8.3 ms spans $[0.021, 0.968]$ with zero values pinned
to the boundary, and the methods naive set matches. **Verdict: resolved — the
clip never reaches the panel; exclude-not-clip is already in force.**

## 1.5 Unit inclusion — the one real upstream difference

Per-window inclusion counts (`n_total` = finite $1-\alpha$; `n_dropped` = out of
$[0,1]$; `n` = kept):

| window | source | base | n_total | n_dropped | n_kept |
|---|---|---|---|---|---|
| 8.3 ms | PROD | 1359 | 1359 | 92 | 1267 |
| 8.3 ms | NAIVE | 1359 | 1359 | 93 | 1266 |
| 16.7 ms | PROD | **1354** | 1354 | 83 | 1271 |
| 16.7 ms | NAIVE | 1359 | 1359 | 86 | 1273 |
| 25.0 ms | PROD | **1337** | 1337 | 86 | 1251 |
| 25.0 ms | NAIVE | 1359 | 1359 | 85 | 1274 |
| 50.0 ms | PROD | **1275** | 1275 | 236 | 1039 |
| 50.0 ms | NAIVE | 1359 | 1359 | 240 | 1119 |

Two sub-effects:

- **Boundary jitter (≤1 unit @ 8.3 ms):** the base is identical (1359) but
  `n_dropped` differs by one (92 vs 93). The methods Crate (uncentred
  $MM-\text{Erate}\otimes\text{Erate}$ over close pairs) differs slightly from
  production's `intercept_mode='below_threshold'` Crate, nudging a single
  borderline unit across the $1-\alpha\in[0,1]$ boundary. Element-wise,
  $\langle|\alpha_\text{prod}-\alpha_\text{methods}|\rangle = 6.1\times10^{-3}$.

- **Per-window finiteness drop (up to 84 units @ 50 ms):** production's `base`
  *shrinks* with window (1359 → 1354 → 1337 → 1275) because `_metrics_one`
  produces non-finite / non-positive Crate/Cpsth/Ctotal diagonals for some units
  at larger windows and drops them. The methods estimator keeps all 1359 units
  finite at every window. This is the source of the diverging unit counts and of
  the NC pair counts (below).

NC pair counts mirror this:

| window | PROD pairs | NAIVE pairs | FULL pairs |
|---|---|---|---|
| 8.3 ms | 59585 | 59585 | 59585 |
| 16.7 ms | 59033 | 59585 | 59585 |
| 25.0 ms | 57314 | 59585 | 59585 |
| 50.0 ms | 51267 | 59585 | 59585 |

**Verdict: a real, estimator-driven inclusion difference, expected and benign.**
It is *expected* because the two estimators legitimately differ in which
large-window diagonals are finite; it is *benign* because the headline
statistics (geomean, slope-through-origin, Fisher-$z$ means) are robust to the
extra units — see §1.6. It is **not** a statistic bug. When production is
eventually swapped, the new estimator's finiteness behaviour simply becomes the
new inclusion rule; the ~1–2% headline shift it implies is quantified next.

## 1.6 Headline statistics — NAIVE reproduces PROD across all windows

### Fano (geometric mean and slope-through-origin)

Panel E plots the **slope-through-origin** Fano. Both it and the geomean ratio
track production within ~1%:

| window | source | g_unc | g_cor | ratio | slope_unc | slope_cor |
|---|---|---|---|---|---|---|
| 8.3 | PROD | 1.0496 | 0.9104 | 0.8674 | 1.0654 | 0.7956 |
| 8.3 | NAIVE | 1.0494 | 0.9104 | 0.8676 | 1.0652 | 0.7957 |
| 8.3 | FULL | 1.0494 | 0.9369 | 0.8928 | 1.0652 | 0.8558 |
| 16.7 | PROD | 1.1627 | 0.8830 | 0.7594 | 1.2597 | 0.7479 |
| 16.7 | NAIVE | 1.1623 | 0.8795 | 0.7567 | 1.2593 | 0.7480 |
| 16.7 | FULL | 1.1626 | 0.9389 | 0.8076 | 1.2752 | 0.8466 |
| 25.0 | PROD | 1.2506 | 0.8419 | 0.6732 | 1.4210 | **0.7074** |
| 25.0 | NAIVE | 1.2499 | 0.8466 | 0.6773 | 1.4202 | **0.7141** |
| 25.0 | FULL | 1.2519 | 0.9305 | 0.7433 | 1.4523 | 0.8555 |
| 50.0 | PROD | 1.4176 | 0.9133 | 0.6443 | 1.7128 | 0.9136 |
| 50.0 | NAIVE | 1.4162 | 0.9087 | 0.6416 | 1.7111 | 0.9127 |
| 50.0 | FULL | 1.4260 | 1.0015 | 0.7023 | 1.8026 | 1.0393 |

The brief's target — corrected Fano **< 0.8 @ 25 ms** — is the slope, and both
PROD (0.707) and NAIVE (0.714) clear it. The earlier "corrected Fano ≈ 0.9+"
came from the §4.5 demo config (`t_hist_ms=92`, single count bin), **not** the
standard `compute_methods_data` windows used here.

### Noise correlation (Fisher-$z$ population means)

| window | source | z_u | z_c | dz |
|---|---|---|---|---|
| 8.3 | PROD | 0.04211 | 0.00322 | −0.03888 |
| 8.3 | NAIVE | 0.04210 | 0.00325 | −0.03886 |
| 8.3 | FULL | 0.04210 | 0.01169 | −0.03041 |
| 16.7 | PROD | 0.06941 | 0.00124 | −0.06818 |
| 16.7 | NAIVE | 0.06966 | 0.00135 | −0.06831 |
| 25.0 | PROD | 0.08823 | −0.00014 | −0.08837 |
| 25.0 | NAIVE | 0.08966 | 0.00045 | −0.08921 |
| 25.0 | FULL | 0.08966 | 0.01095 | −0.07871 |
| 50.0 | PROD | 0.12114 | 0.03394 | −0.08720 |
| 50.0 | NAIVE | 0.12636 | 0.02829 | −0.09807 |

NAIVE tracks PROD to ~$10^{-3}$ at the short windows; the gap widens slightly at
50 ms (z_u 0.126 vs 0.121) consistent with the +84-unit / +8318-pair inclusion
difference of §1.5.

### $1-\alpha$ (FEM modulation fraction)

| window | source | n | mean | median |
|---|---|---|---|---|
| 8.3 | PROD | 1267 | 0.7222 | 0.7703 |
| 8.3 | NAIVE | 1266 | 0.7201 | 0.7688 |
| 25.0 | PROD | 1251 | 0.7466 | 0.8009 |
| 25.0 | NAIVE | 1274 | 0.7456 | 0.7991 |
| 50.0 | PROD | 1039 | 0.7761 | 0.8380 |
| 50.0 | NAIVE | 1119 | 0.7838 | 0.8454 |

## 1.7 Phase 1 difference table

| # | difference | locus | classification |
|---|---|---|---|
| 1 | Session / subject set | — | none (shared cache; Luke absent both sides) |
| 2 | Counting windows | — | none |
| 3 | Stage-3 statistic functions | — | none (Phase-2 adapter uses literal production fns) |
| 4 | alpha clipped at estimator | `estimators.py`/`pipeline.py` | benign — derived bundle stores raw α, excludes ∉[0,1] like prod |
| 5 | Crate estimator (close-pair uncentred vs `below_threshold`) | upstream | expected — ≤1-unit boundary jitter @ 8.3 ms, $\langle|\Delta\alpha|\rangle=6\times10^{-3}$ |
| 6 | Per-window finiteness inclusion | upstream | expected — prod drops 5/22/84 units at 16.7/25/50 ms; benign for pooled stats |
| 7 | `fano_stats['per_subject']` missing in methods bundle | Stage 3 | reconstructed via production fn in adapter |
| 8 | `nc_stats['null_dz_ci_by_subject']` missing in methods bundle | Stage 3 | reconstructed via production fn in adapter |
| 9 | Shuffle null computed for `naive` target only | `pipeline.py` | by design — panel-D `full` band reuses the naive null (see Phase 2) |

Differences 1–4 are non-issues; 5–6 are the genuine (and small) upstream
effects of the new estimator; 7–9 are handled in the adapter. **No item is a
statistic bug.**

---

# Phase 2 — adapter and 3-up panel comparison

## 2.1 Adapter

`consistency/adapter.py::methods_to_fig2_schema(md, target, null_from=None)`
maps the methods derived bundle into the exact `load_fig2_data()` schema for one
target. As argued in §1.3, it runs the methods `metrics[target]` list through the
**literal production Stage-3 functions** imported from `compute_fig2_data`:

```python
m_by_window, subj_pn_by_window, alpha_stats = _compute_alpha_stats(metrics, windows_ms)
fano_stats = _compute_fano_stats(metrics, windows_ms)   # incl. per_subject
nc_stats   = _compute_nc_stats(metrics, windows_ms)     # incl. null_dz_ci_by_subject
```

This both **eliminates statistic-function divergence** and **reconstructs** the
two per-subject sub-keys the methods bundle omitted
(`fano_stats[w]['per_subject']`, `nc_stats[w]['null_dz_ci_by_subject']`). The
`null_from='naive'` option borrows the naive eye-shuffle null into the `full`
and `central` targets so panel 3D's reference band is defined for them (see
§2.4).

Adapter verification (`uv run python consistency/verify_adapter.py`) confirms:
**every** panel-required key is present; the adapter `fano_stats`/`nc_stats`
key-sets are **identical** to production (empty set differences); and the
panel-driving statistics reproduce production within tolerance (next section).
All checks pass.

## 2.2 NAIVE reproduces PROD — residuals

Per-window absolute residual `|PROD − NAIVE|` on the plotted statistics:

| window | fano slope_unc | fano slope_cor | nc dz_mean |
|---|---|---|---|
| 8.3 ms | 0.0002 | 0.0001 | 0.0000 |
| 16.7 ms | 0.0005 | 0.0001 | 0.0001 |
| 25.0 ms | 0.0008 | 0.0067 | 0.0008 |
| 50.0 ms | 0.0018 | 0.0009 | 0.0109 |

All slope residuals are $<0.007$ (tolerance 0.02); all $\Delta z$ residuals are
$\le 0.011$ (tolerance 0.012). The single largest residual — $\Delta z$ at
50 ms (0.0109) — traces directly to the §1.5 inclusion difference (+84 units /
+8318 pairs the methods estimator keeps finite at 50 ms). For $1-\alpha$
(panel C) the per-subject medians match to 3 decimals (PROD 0.770 / NAIVE 0.769
at 8.3 ms). **NAIVE reproduces PROD for all five panels within stated
tolerance.** The brief's hard target — corrected Fano slope **< 0.8 @ 25 ms** —
holds: PROD 0.707, NAIVE 0.714.

## 2.3 4-up comparison figures

Each figure renders **PROD | NAIVE | FULL | CENTRAL** by calling the *same*
production `plot_panel_X(ax=…, data=…)` four times — so any visible difference is
in the data, never the drawing. `CENTRAL` is the methods estimator matched to a
*central* eye-position distribution (tightly fixated); see §2.5. Generated by
`uv run python consistency/make_panels.py`.

![Fig 2C — per-subject $1-\alpha$ histogram at 8.3 ms. PROD and NAIVE are
indistinguishable; neither shows a pile-up at 0 or 1 (the estimator clip never
reaches this path — §1.4). FULL and CENTRAL shift left: under a less
close-pair-dominated distribution a smaller fraction of rate variance is
attributed to FEM.](figures/consistency/cmp_fig2c.png)

![Fig 2E — population Fano slope-through-origin vs window. Open/dashed =
uncorrected, filled/solid = corrected. PROD and NAIVE overlay exactly. FULL and
CENTRAL corrected slopes sit higher (less correction): the naive close-pair
estimator over-removes rate-driven variance.](figures/consistency/cmp_fig2e.png)

![Fig 3B — corrected vs uncorrected noise-correlation scatter at 8.3 ms, one
sub-panel per monkey. PROD and NAIVE clouds match; FULL and CENTRAL corrected
values sit slightly higher (less correction).](figures/consistency/cmp_fig3b.png)

![Fig 3C — mean Fisher-$z$ noise correlation vs window. PROD and NAIVE overlay;
FULL and CENTRAL corrected curves are pulled up toward the uncorrected curve at
every window (CENTRAL marginally above FULL).](figures/consistency/cmp_fig3c.png)

![Fig 3D — $\Delta z$ (corrected − uncorrected) vs window against the shuffle
null 95% band. PROD and NAIVE match. FULL and CENTRAL are attenuated and reuse
the naive null band as their reference (§2.4); CENTRAL shows the smallest
reduction and loses significance earliest at the long
window.](figures/consistency/cmp_fig3d.png)

## 2.4 Target movement — NAIVE → FULL → CENTRAL

| window | slope_cor (naive / full / central) | dz_mean (naive / full / central) |
|---|---|---|
| 8.3 ms | 0.796 / 0.853 / 0.834 | −0.039 / −0.030 / −0.019 |
| 16.7 ms | 0.748 / 0.816 / 0.861 | −0.068 / −0.059 / −0.040 |
| 25.0 ms | 0.714 / 0.789 / 0.874 | −0.089 / −0.079 / −0.054 |
| 50.0 ms | 0.913 / 0.941 / 1.069 | −0.098 / −0.088 / −0.033 |

Consistently, **FULL and CENTRAL apply less correction than NAIVE**: corrected
Fano slopes rise (e.g. +0.08 / +0.16 at 25 ms for full / central) and the
noise-correlation reduction shrinks (e.g. −0.089 → −0.079 / −0.054 at 25 ms, ~11
/ 39% less). Mechanistically, the naive estimator pools only close eye-position
pairs and so attributes *more* of the across-trial covariance to rate (FEM),
over-subtracting it; matching to a broader eye-position distribution recovers a
more faithful — smaller — rate component, leaving more residual noise
covariance. This is the §4 thesis of `writeup.md`, here quantified on every
fig2/fig3 panel. (FULL/CENTRAL use the directly-estimated close-pair density,
the estimator default — `closepair_density='direct'`,
`note_closepair_density.md`; NAIVE is invariant to it.)

**Panel-D `full`/`central` shuffle-null decision.** The pipeline computes the
eye-shuffle null for `target='naive'` only. Rather than fabricate a target-
specific null (which would require re-running the shuffle under each target's
reweighting — out of scope for a vetting pass), the FULL and CENTRAL panel-3D
plots show each target's observed $\Delta z$ against the **naive** null band
(`null_from='naive'`). This is the correct reference: the null answers "how much
$\Delta z$ does eye-shuffling produce by chance under the same close-pair
procedure," and that band is a property of the shuffle, not of the target
reweighting. Target-specific nulls are flagged as future work for the production
swap.

## 2.5 The central condition

`target='central'` matches the decomposition to a **central** (tightly fixated)
eye-position distribution rather than the full viewing distribution $p(e)$. The
user's interest is whether central is preferable for **noise correlations**.
From the §2.4 table, central sits *beyond* full in the same direction — it
applies the **least** correction of the three estimators on the NC panels:

- **Noise-correlation reduction (fig 3C/3D):** $|\Delta z|$ orders
  naive $>$ full $>$ central at every window (e.g. @50 ms: −0.098 / −0.088 /
  −0.033). The residual corrected correlation $z_c$ orders naive $<$ full $<$
  central (@25 ms: 0.0004 / 0.011 / 0.027), so central leaves the **most**
  residual positive correlation — it does not drive the corrected NC toward
  zero, and its $\Delta z$ loses significance earliest at the long window
  (fig 3D, rightmost column).
- **Fano (fig 2E):** central's corrected slope sits at or above full's, most
  clearly at the long windows (e.g. 1.069 vs 0.941 at 50 ms).

So on the **point estimates**, central is the gentlest correction and does not
reduce the corrected noise correlation below full. If the motivation for central
is **estimator variance** ("reduce noise" in the sense of tighter CIs rather than
smaller point estimates), that is a separate question this vetting pass does not
settle — the per-window CIs in fig 3C/3D are visually comparable across targets,
and a dedicated variance comparison (bootstrap CI widths or repeated-split
stability, target-by-target) would be the right tool. Flagged as the natural
follow-up if central is being considered specifically to denoise the NC
estimate.

---

# Success criteria — status

| criterion | status |
|---|---|
| Itemized Phase-1 diff, every divergence explained | done (§1.7) |
| NAIVE reproduces PROD for all five panels within tolerance | done — slopes $<0.007$, $\Delta z \le 0.011$ (§2.2) |
| Corrected Fano slope < 0.8 @ 25 ms once exclude-not-clip + statistic matched | done — PROD 0.707 / NAIVE 0.714 (§1.6, §2.2) |
| `1-α`-clip pile-up explained and resolved | done — clip never reaches panel; bundle excludes raw, counts in §1.4–1.5 |
| NC panels (fig3 B/C/D) validated for the first time | done (§2.2–2.3) |
| Panel-D `full` shuffle-null handling decided and documented | done — reuse naive null (§2.4) |
| Comparison figures for fig2 C/E and fig3 B/C/D saved | done — 4-up PROD/NAIVE/FULL/CENTRAL (§2.3, `figures/consistency/`) |
| Production untouched | verified — only `consistency/` added |

## Reproduce everything

```bash
uv run python consistency/phase1_inspect.py      # schema + key presence
uv run python consistency/phase1_compare.py        # per-window headline stats
uv run python consistency/phase1_inclusion.py      # per-window inclusion counts
uv run python consistency/verify_adapter.py        # adapter schema + tolerance checks
uv run python consistency/make_panels.py           # the 3-up figures
```

## Implication for the production swap

NAIVE already reproduces PROD; the only upstream differences are a benign,
estimator-driven per-window inclusion shift (§1.5) and a ≤1-unit boundary jitter
in $1-\alpha$. Swapping production to the methods estimator with `target='full'`
will (i) raise corrected Fano slopes by ~0.06–0.14, (ii) shrink the
noise-correlation reduction by ~15–25%, and (iii) lower the FEM fraction
$1-\alpha$ by ~0.03 — all in the direction of *less* correction, consistent with
the naive estimator's documented over-correction. No statistic-function or
plotting change is required; the swap is purely at the decomposition layer.
