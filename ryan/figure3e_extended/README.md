# figure3e_extended — the stabilization ladder

A fork of figure 3's single-trial panel, expanded from three within-model
conditions to six and from one metric to five. Self-contained: it owns its cache
and imports nothing from `paper/fig3` (it does read fig2's decomposition for the
per-unit ceiling in row B).

## What is new relative to figure 3

Figure 3 contrasts the intact twin against one reafferent ablation (retinal
input frozen at a single session-global gaze). That is the *broadest* possible
stabilization, and it conflates two things: removing retinal image motion, and
placing every trial at the same retinal position. This fork separates them by
sweeping the **scope** over which the retinal input is stabilized:

| condition | retinal input | behavior | colour |
|---|---|---|---|
| `psth` | — (leave-one-out trial average) | — | grey |
| `full` | intact | intact | blue |
| `ablated` | intact | zeroed | red |
| `stab_window` | stabilized inside the model's 33-frame window | intact | purple |
| `stab_trial` | frozen at each trial's own centroid gaze | intact | purple |
| `stab_global` | frozen at one session-global centroid gaze | intact | purple |
| `stab_global_ablated` | frozen at the session-global gaze | zeroed | green |

`stab_global` reproduces figure 3's `stabilized` condition, and the renderer is
gated to be bit-exact against fig3's production code, so the fork is anchored to
a known number.

Five quantities are computed per neuron (the Poisson ones in `_ext_metrics.py`):

- **single-trial `r^2`** against the observed spike counts
- **fraction of explainable variance**, `r^2 / R^2_max` — the same `r^2`
  normalized by the ceiling fig2 measures for that unit. See below.
- **single-trial Poisson bits per spike** against each unit's own mean-rate null
- **Poisson self-consistency fraction** — see below
- **residual rate fraction**, `var(rate_full − rate_perturbed) / var(rate_full)`
  — how much of the twin's *own* predicted rate modulation each perturbation
  moves. This one is model-vs-model: the observed spikes play no part, so it
  measures what the twin's rate depends on rather than how well it predicts.
  It is computed on the **raw** model output, not the affine-rescaled rates,
  because the rescaling is fit per condition against the observed counts and
  would partly absorb the very change being measured. A pure DC shift between
  conditions cancels inside a variance, so this reflects changed *modulation*,
  not a changed mean rate. The rescaled variant is cached as
  `resid_frac_rescaled` for comparison. The fraction can exceed 1: a perturbed
  twin is not a shrunken full twin, so the difference can carry more variance
  than the reference.

## The explainable-variance fraction

Row A's `r^2` has no absolute scale: `0.039` says nothing about how much of the
spiking was predictable in the first place. Fig. 2 already measured that, per
unit. Decomposing the observed counts as `y = lambda + noise`, where
`lambda = E[y | stimulus, gaze]`,

```
Var(y) = Var(lambda) + E[Var(y|z)]
```

and any predictor that is a function of the conditioning variables satisfies
`E[(y - yhat)^2] = E[(lambda - yhat)^2] + E[Var(y|z)] >= E[Var(y|z)]`, with
equality only at `yhat = lambda`. So

```
r^2  <=  Var(lambda) / Var(y)  =  diag(Crate) / diag(Ctotal)  =:  R^2_max
```

This is exact. It assumes only that the residual is conditionally mean-zero
given the stimulus and gaze — not that it is Poisson. Both terms are already in
the stage-1 decomposition: `Ctotal` is the sample covariance of windowed counts,
and `Crate` is the close-pair estimator, importance-reweighted (`target='full'`)
so the rate variance is measured under the actual viewing distribution — the
same distribution this `r^2` averages over, which is what makes the ratio a
fraction of anything. `paper/covariance_decomposition/ceiling.py` reshapes those
diagonals into a small `(session, neuron_id)`-keyed table so this figure never
opens the 8 GB stage-1 cache.

**The window is not a free choice.** `R^2_max` rises steeply with the counting
window (median `0.119` at 1 bin, `0.279` at 3) because summing counts averages
private noise down faster than rate variance. The twin runs at 120 Hz — the
dataset configs decimate 240 -> 120, and `factor = 2` is that decimation applied
to stimulus frames, not a binning of counts — so the ceiling is taken at
`window_bins = 1`, matching the bin the `r^2` is computed on.

Reading it: the full twin's median is `0.35`, i.e. it captures about a third of
the variance that is capturable in principle; `stab_global` sits at `0.02`.
Because the denominator is estimated from data alone and is *identical across
all seven conditions*, it cannot reward a condition for predicting less — the
failure mode of the self-consistency row. The flip side is that within a unit
it is a pure rescaling and does not reorder the conditions; what it adds is the
absolute scale and the removal of across-unit noise heterogeneity before
pooling.

Three caveats:

1. **`R^2_max` is a lower bound on the true ceiling.** Close pairs are matched
   on a trajectory spanning `t_hist + t_count` bins while the twin conditions on
   a 33-frame history; to the extent `lambda` depends on gaze history beyond the
   matched window, or varies within the 0.05 deg tolerance, paired samples do
   not share a rate and `Crate` is under-estimated. So `r^2 / R^2_max > 1` is
   possible and the line at 1 is not a hard bound.
2. **Not gated on the eye-shuffle null.** `p_rate` is carried in the table, but
   the shuffle permutes trajectories while leaving `T_idx` alone and close pairs
   are still enumerated within a time-in-trial group, so a shuffled `Crate`
   retains the PSTH variance — that null tests for resolvable *FEM* modulation
   (it is the null fig2 uses for `1 - alpha`), not for usable rate variance.
   Gating this row on it would drop 294 of 972 cells and quietly restrict the
   row to FEM-modulated units, which is not what a fraction of *total*
   explainable variance describes. The only exclusion is a non-positive or
   non-finite denominator: 22 of 994 population cells.
3. **Undefined, not clipped**, in both the ceiling table and this row, so the
   pile-up a clip would create at the bounds never appears.

One inherited bias: `_var_explained` is `1 - nanvar(pred - true)/nanvar(true)`,
a residual *variance* rather than a mean square, so a non-zero residual mean
goes unpenalized. The affine rescaling is fit on Poisson deviance, whose
stationarity condition is `sum(y/q) = N` rather than `sum(y - q) = 0`, so the
residual mean is small but not exactly zero. This inflates every `r^2` on the
figure slightly, row A included; it is pre-existing and not introduced by the
normalization, and quantifying it needs the per-bin traces the cache does not
keep.

## Within-window stabilization

The narrowest condition. For every prediction time `t` and lag `l` in 0..32:

```
frame(t, l) = crop( image on screen at time t-l,  gaze ROI at time t )
```

Each lag keeps the image that was really on screen, so the RSVP flash dynamics
are untouched; but the whole window is cropped at the *last* frame's gaze, so no
eye movement survives inside the window. Across prediction times the crop still
follows the eye, so gaze-contingent position shifts remain. It removes retinal
image motion and nothing else — the strictest reafference control available, and
unlike trial-wise stabilization it is definable on non-repeated stimuli.

### How it is rendered

A stabilized *copy of the stimulus movie* cannot express this: the frame at a
given time is no longer well defined, since it depends on which prediction
window it belongs to. The cube has to be built per prediction time, which is
~4,000 frames per trial — hopeless through `FixRsvpTrial.get_rois`, which
regenerates a full image texture per frame.

The way out is that `roi = dpi_pix.astype(int) + roi_src` is a pure integer
translation of gaze, and `place_gauss_image_texture` pastes through `nd_paste`,
which is translation-covariant with a constant fill outside the source. So two
crops of the same screen content at two different gazes are two slices of one
canvas. Each distinct screen *content* — image identity, draw position,
background, radius — is therefore rendered once into a canvas spanning the
screen, and every frame the model sees is a numpy slice. A session has ~30
distinct contents, so this is ~30 renders per session instead of ~500 per trial:
about 200x faster than the route fig3 currently uses for its stabilized render.

Because content is resolved per sample rather than per trial, a lag reaching back
across a trial boundary (37% of fixation samples at a 33-frame window) renders
the *previous* trial's image at the current gaze — exactly what the intact window
holds, minus the eye motion. No clamping, no boundary special case.

### Gates (fatal per session)

1. `alignment_maxabs == 0` — decimating the stored raw stimulus reproduces the
   embedded model stimulus bit-exactly, so raw index = `factor * model index`.
2. `verify_native_render == 0` — slicing each canvas at its own native ROI
   reproduces the stored raw stimulus bit-exactly, over precisely the samples
   this session will crop at. Proves canvas-slicing == `get_rois`, that the
   content reconstruction matches the stored stimulus, and that the re-render
   carries no artifact.
3. `stab_global` is separately checked bit-exact against
   `paper/fig3/_fig3_ablation_data.build_stabilized_stim`.

## The Poisson self-consistency fraction

Bits per spike is a principled score with an unfamiliar absolute scale: nothing
in `0.13 bits/spike` says how close that is to the score a *correct* stochastic
encoding model would earn. This row supplies that reference, per condition.

For observed counts `y_i`, predicted mean counts `q_i` and the mean-rate null
`q0_i`, the observed likelihood gain over the null is

```
G_obs  = Σ_i [ y_i log(q_i/q0_i) − q_i + q0_i ]
```

which is exactly the numerator bits/spike divides by `Σ_i y_i · log 2`. If the
predicted rates were correct and everything left over were independent Poisson
noise, `Y_i ~ Poisson(q_i)`, then `E[Y_i] = q_i` and — since `G` is affine in
`y` — the expected gain has a closed form,

```
G_self = Σ_i [ q_i log(q_i/q0_i) − q_i + q0_i ]
       = Σ_i KL[ Poisson(q_i) ‖ Poisson(q0_i) ] ≥ 0
```

with equality only when the prediction *is* the null in every valid bin. Their
ratio `S = G_obs / G_self` has expectation 1 when `q` is the true conditional
Poisson mean. No sampling is involved; the analytic expression is what runs, and
a seeded Monte Carlo check of it lives in `tests/test_fig3e_metrics.py`.

Reading it: `S ≈ 1` means the spikes deliver the gain the model claims under its
own rates; `0 < S < 1` means they deliver less; `S < 0` means the null predicts
better. **`S > 1` is possible** — the denominator is an expectation, not a hard
bound, so this is not a ceiling and should not be drawn as one.

Three caveats, all structural:

1. **Each condition has its own denominator**, so `S` does *not* rank conditions
   by total predictive information. A weak prediction can score near 1 by
   claiming little and delivering it. Read this row together with row B
   (bits/spike) — a condition can lose there either by predicting less
   modulation (lower `G_self`) or by predicting modulation the spikes do not
   support (lower `S`), and only the pair separates the two.
2. **A near-null prediction has a near-zero denominator.** The fraction is
   reported as `NaN` when `G_self` falls below `1e-12 ×` the total predicted
   count, rather than clipped to some finite value.
3. **This is not cross-fitted.** The affine rescaling and the mean-rate null are
   both estimated on the responses being scored, so the expected-one result
   holds only up to that in-sample calibration. Cross-fitting them would move
   the bits/spike row too, so comparability with row B was kept instead; that
   decision is deliberate and is recorded in `_ext_metrics.py`.

The masking, the positive-rate floor, and the mean-rate null are shared with the
bits/spike row bin-for-bin, and a unit test asserts `G_obs / (N log 2)`
reproduces `calc_poisson_bits_per_spike` exactly. `gain_obs` and `gain_self` are
cached per neuron alongside the ratio.

`poisson_self_consistency_report.tex` carries the full derivation, the sampling
variance, and the relation to deviance-explained and repeated-trial ceilings.

## Two notes on the other metrics

**The PSTH is rescaled like every other condition.** The raw leave-one-out PSTH
has structural zeros (bins where no other trial happened to spike), and a Poisson
likelihood charges `log(0)` whenever the held-out trial spikes there, so an
unrescaled PSTH scores a large negative bits/spike that describes the estimator's
zeros rather than the trial average's predictive power. Every predictor,
including the PSTH, therefore goes through the same affine rescaling
(`rescale_rhat`, mode `affine`), whose exponentiated parameterisation is strictly
positive by construction. This *strengthens* the baseline — the conservative
direction — but it means the PSTH `r^2` here is not the same number as fig3's
unrescaled PSTH reference line. The unrescaled value is kept per session as
`ve_psth_unrescaled` and printed at run time.

**Population.** Identical to fig2/fig3: rate > 2 Hz and split-half PSTH
`R^2 > 0.10`, with fig2's 10-analyzed-unit session floor. Both rules are
recomputed here from the 72 MB aligned covariance cache rather than the multi-GB
derived bundle, and reproduce it exactly (19 sessions, 11 Allen / 8 Logan,
1022 units).

## Files

- `_ext_stim.py` — the canvas renderer, the three stabilization scopes, the gates
- `_ext_metrics.py` — the single-trial Poisson metrics (bits/spike and the
  self-consistency fraction), importable without the renderer, the data packages
  or a GPU checkpoint
- `_ext_data.py` — six-condition inference, the per-neuron quantities, the
  ceiling join (`_attach_ceiling`), owns
  `outputs/cache/fig3e_extended_ablation.pkl`
- `generate_figure3e_extended.py` — the five-row figure
- `poisson_self_consistency_report.tex` — derivation and limitations of row E
- `../../paper/covariance_decomposition/ceiling.py` — the per-unit `R^2_max`
  table (row B); owns `outputs/cache/covdecomp_ceiling.pkl`
- `../../tests/test_fig3e_metrics.py` — unit tests for `_ext_metrics.py`
- `../../tests/test_ceiling.py` — join key, undefined-not-clipped rule, and the
  shuffle-null p for the ceiling table

Note the cache stores per-neuron summaries only, not the per-bin rate traces, so
any new trace-derived quantity (like the residual and self-consistency rows were)
costs a full re-run of the sweep. Loading a cache that predates a row prints a
`NOTE:` from `aggregate()`, and the figure then fails with an explicit
`--recompute` instruction rather than a bare `KeyError`.

## Usage

```bash
# full sweep (~70 min on a shared GPU; 24 sessions x 6 conditions) + figure
uv run python ryan/figure3e_extended/generate_figure3e_extended.py --recompute

# figure only, from cache
uv run python ryan/figure3e_extended/generate_figure3e_extended.py

# rebuild the per-unit ceiling table (row B) after a fig2 re-decomposition;
# loads the 8 GB stage-1 cache once, writes a ~1 MB table
uv run python paper/covariance_decomposition/ceiling.py --refresh

# one session, off the real cache
uv run python ryan/figure3e_extended/_ext_data.py \
    --sessions Allen_2022-02-16 --cache /tmp/smoke.pkl
```

Outputs land in `outputs/figures/fig3e_extended/`.
