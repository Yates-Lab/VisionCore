# figure3e_extended — the stabilization ladder

A fork of figure 3's single-trial panel, expanded from three within-model
conditions to six and from one metric to two. Self-contained: it owns its cache
and imports nothing from `paper/fig3`.

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

Three quantities are computed per neuron:

- **single-trial `r^2`** against the observed spike counts
- **single-trial Poisson bits per spike** against each unit's own mean-rate null
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

## Two notes on the metrics

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
- `_ext_data.py` — six-condition inference, `r^2` + bits/spike, owns
  `outputs/cache/fig3e_extended_ablation.pkl`
- `generate_figure3e_extended.py` — the three-row figure

Note the cache stores per-neuron summaries only, not the per-bin rate traces, so
any new trace-derived quantity (like the residual row was) costs a full re-run of
the sweep.

## Usage

```bash
# full sweep (~70 min on a shared GPU; 24 sessions x 6 conditions) + figure
uv run python ryan/figure3e_extended/generate_figure3e_extended.py --recompute

# figure only, from cache
uv run python ryan/figure3e_extended/generate_figure3e_extended.py

# one session, off the real cache
uv run python ryan/figure3e_extended/_ext_data.py \
    --sessions Allen_2022-02-16 --cache /tmp/smoke.pkl
```

Outputs land in `outputs/figures/fig3e_extended/`.
