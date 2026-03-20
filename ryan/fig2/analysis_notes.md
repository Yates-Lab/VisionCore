# Figure 2: Covariance Decomposition — Analysis Notes

## Overview

Figure 2 decomposes total spike-count covariance into stimulus-locked (PSTH),
fixational eye movement (FEM), and intrinsic noise components using the law of
total covariance. The headline finding: FEMs account for ~80% of rate variance,
and noise correlations collapse from r ~ 0.05 to ~ -0.01 after FEM correction.

**Figure file:** `fem-v1-fovea/figures/fig2_covariance_decomposition.pdf`

## Key Files

| File | Purpose |
|------|---------|
| `ryan/figure_lotc_model.py` | Original figure generation script (rushed, monolithic) |
| `scripts/mcfarland_sim.py` | Core analysis: `DualWindowAnalysis`, `run_mcfarland_on_dataset()`, `extract_metrics()` |
| `scripts/figures_fanofactors.py` | Fano factor stats and plotting |
| `scripts/figures_noisecorr.py` | Noise correlation stats and plotting |
| `scripts/figures_alpha.py` | Alpha (FEM modulation fraction) stats and plotting |
| `scripts/figures_subspace.py` | Subspace alignment stats and plotting |
| `scripts/figure_main_lotc.py` | `create_main_figure()` — unified 3x3 panel layout (panels C-K) |
| `scripts/figure_common.py` | Shared utilities (bootstrap, IQR, Fisher z, pub style) |
| `scripts/utils.py` | `get_model_and_dataset_configs()` |

## Data Pipeline

### Source
- **Species:** Common marmoset (Callithrix jacchus)
- **Brain area:** V1 (foveal representation)
- **Recording:** Multi-electrode arrays (Yates lab: 64-ch tungsten; Rowley: Neuropixels)
- **Eye tracking:** DPI (Dual Purkinje Image)
- **Paradigm:** fixRSVP — rapid serial visual presentation during fixation
- **Sessions:** 20 sessions across marmosets Allen and Logan

### Loading
1. `get_model_and_dataset_configs(mode="standard")` loads a pre-trained multi-dataset model
2. `run_mcfarland_on_dataset(model, dataset_idx)` loads train/val data via `load_single_dataset()`
3. Only fixRSVP trials are selected: `train_data.get_dataset_inds('fixrsvp')` + val equivalent
4. Data shapes: `robs` (NT, T, NC), `eyepos` (NT, T, 2)

### Pickle cache
- `dmcfarland_outputs_standard.pkl` — list of per-session output dicts
- `dmcfarland_analyzers_standard.pkl` — list of DualWindowAnalysis objects

## Current Inclusion Criteria

### What exists
1. **Total spike threshold:** neurons with < 200 total spikes excluded (`run_mcfarland_on_dataset`, line 2549)
2. **Fixation duration:** trials with < 20 valid time bins excluded (line 2525)
3. **Metrics extraction:** secondary filter at `min_total_spikes=500`, `min_var=0` (line 180 of figure script)
4. **Variance floor:** neurons with diagonal variance <= `min_var` excluded during correlation computation

### What does NOT exist
- No minimum firing rate per neuron
- No isolation quality filter (Kilosort4 provides good/mua/noise labels — unused)
- No SNR or RF quality criteria (available in DataYatesV1 — unused)
- No laminar position filters
- No stationarity checks
- No rate-matching across pairs

## Covariance Decomposition Method

### Law of Total Covariance (applied twice)

```
Σ_total = Σ_PSTH + Σ_FEM + Σ_int
```

Where:
- **Σ_total**: standard sample covariance of raw spike counts (Bessel-corrected)
- **Σ_rate** = Σ_PSTH + Σ_FEM: estimated by conditioning on eye trajectory similarity
- **Σ_PSTH**: bagged split-half cross-covariance of time-locked mean responses
- **Σ_FEM** = Σ_rate - Σ_PSTH: residual rate covariance attributed to eye movements
- **Σ_int** = Σ_total - Σ_rate: intrinsic noise (what remains after accounting for rate)

### Σ_rate estimation (eye-trajectory matching)
1. Extract non-overlapping windows of spike counts and eye trajectories
2. Compute RMS distance between all pairs of eye trajectory windows
3. Bin pairs by trajectory distance
4. For each bin, compute raw second moment E[y_i y_j^T | d] (distinct trials only)
5. Fit monotonic regression (PAVA) across bins → intercept at d→0
6. Convert second moment to covariance: Σ_rate = MM_intercept - E[rate] * E[rate]^T

### Σ_PSTH estimation (bagged split-half)
1. K=20 bootstrap iterations
2. Each iteration: randomly split trials 50/50
3. Compute mean response per time bin on each half
4. Cross-covariance of the two halves (unbiased: avoids noise-on-noise inflation)
5. Average across bootstrap iterations

### Derived quantities
- **Noise correlations (uncorrected):** correlations of Σ_total - Σ_PSTH
- **Noise correlations (corrected):** correlations of Σ_total - Σ_rate
- **Fano factor (uncorrected):** diag(Σ_total - Σ_PSTH) / E[rate]
- **Fano factor (corrected):** diag(Σ_total - Σ_rate) / E[rate]
- **Alpha:** diag(Σ_PSTH) / diag(Σ_rate) — fraction of rate variance from stimulus
- **1 - alpha:** fraction of rate variance from FEMs (the headline number)

### Window sizes
Analysis run at 10, 20, 40, 80 ms counting windows. Primary claim at 10 ms.
Base temporal resolution: 1/120 s ≈ 8.33 ms bins.

## Shuffle Controls

Eye trajectories permuted across trials while preserving spike counts and stimulus timing.
This breaks the trial-specific eye-spike coupling while preserving marginal statistics.
- n_shuffles = 100 in the current analysis
- Null distributions computed for alpha, Fano ratio, noise correlation delta

## Figure 2 Panels

| Panel | Content |
|-------|---------|
| A | Schematic: LOTC decomposition diagram |
| B | Schematic: eye-trajectory matching method |
| C | Histogram of 1-α (FEM modulation fraction): median 0.802 [0.717, 0.868] |
| D | Mean-variance relationship before/after FEM correction |
| E | Population Fano factor vs counting window (10-80 ms) |
| F | Noise correlation scatter: corrected vs uncorrected (single window) |
| G | Mean Fisher-z noise correlations vs counting window |
| H | Effect size (Δz) vs window with shuffle null band |
| I | Eigenspectra of Σ_PSTH and Σ_FEM |
| J | Participation ratio (effective dimensionality) per session |
| K | Subspace alignment: variance capture X vs Y |

## Cohen & Kohn (2011) Methodological Concerns

Reference: `fem-v1-fovea/references/cohen-kohn-2011/summary.md`

### High severity

1. **Low firing rates bias r_SC toward zero (C&K Fig. 2)**
   - No minimum rate filter. Weakly-driven units depress uncorrected r_SC,
     potentially exaggerating the FEM correction effect.
   - *Control needed:* Rate-threshold sweep; show results hold for neurons > 2, 5, 10 sp/s.

2. **Multiunit contamination inflates r_SC (C&K Fig. 4)**
   - No check for MUA vs SUA. Kilosort4 labels available but unused.
   - *Control needed:* Restrict to "good" isolation units; compare with MUA-included.

3. **Geometric mean rate confound (C&K Fig. 2d)**
   - r_SC depends on minimum rate of the pair, not geometric mean.
   - *Control needed:* Plot r_SC (uncorrected and corrected) vs geometric mean rate.

### Medium severity

4. **Measurement window (C&K Fig. 3)**
   - 10 ms primary window likely underestimates correlations for both conditions.
   - Relative change may be valid, but absolute magnitudes are compressed.
   - *Already partially addressed:* 4 window sizes tested.

5. **Oversorting deflates r_SC (C&K Fig. 5)**
   - No analysis of sort quality vs r_SC.
   - *Control needed:* Check if Kilosort4 aggressiveness affects conclusions.

6. **Internal state / stationarity (C&K discussion)**
   - No drift correction. Long recordings may have slow excitability changes.
   - Shuffle control partially addresses but doesn't fully separate drift from FEM.
   - *Control needed:* Early/late half split; cross-trial correlation (r_LT) analysis.

### Lower severity

7. **Window-size convergence**
   - Show 1-alpha stability across window sizes.

8. **Sub-Poisson Fano factors**
   - Post-correction FF < 1 (0.771 at 10 ms) is a strong claim.
   - Verify not driven by oversorting or low-rate threshold effects.

9. **Pair-distance dependence**
   - C&K emphasize r_SC depends on electrode distance.
   - Show noise correlations vs inter-neuron distance before/after correction.

10. **Signal-noise correlation relationship**
    - Plot r_SC vs r_signal; does FEM correction change this?

## Completed Controls (2026-03-19)

**Script:** `ryan/fig2/generate_figure2_qc_controls.py`
**Figures:** `outputs/figures/fig2_qc/`
**Stats:** `outputs/stats/fig2_qc/fig2_qc_stats.txt`

### QC-stratified robustness analysis (Panels S1-S3)

Tested whether headline findings depend on spike sorting quality using two
QC axes: refractory contamination (%) and fixRSVP-specific missing spike %.

**Results: Spike sorting quality does NOT drive the findings.**
- 1-α (FEM modulation) is flat across contamination range (running median
  0.75–0.80). Well-isolated subset (N=195): median 1-α = 0.746 (vs 0.793 all).
- FF ratio is flat at ~0.88 regardless of QC. Zero dependence on missing %
  (ρ_s = -0.011, p = 0.73).
- Δρ running medians are flat vs contamination and missing %. Well-isolated
  pairs (N=9,515): mean Δρ = -0.085 (vs -0.098 all). Modest 13% reduction.
- Cumulative inclusion curves (Panel S3) show all effects stabilize early and
  do not depend on where the QC threshold is placed.

**Data note:** Missing spike % (truncation analysis) is capped at 50% for ~2/3
of units — a ceiling of the sigmoid fit when amplitude distributions are
centered near detection threshold. Contamination axis is more informative.

### Rate-matched noise correlation analysis (Panel S4)

Tested the dominant confound: firing rate (ρ_s = -0.59 for Δρ vs pair rate).
Cohen & Kohn (2011, Fig. 2) predict threshold masking compresses correlations
toward zero for low-rate neurons.

**Results: FEM correction AMPLIFIES with firing rate.**
- Δz grows from -0.11 (all pairs) to -0.30 (rate ≥ 0.5 sp/bin, N=3,511
  pairs) to -0.95 (rate ≥ 1.0 sp/bin, N=41 pairs).
- Uncorrected z grows from 0.04 to 0.09 (expected threshold unmasking).
- Corrected z goes increasingly negative: -0.06 to -0.20 at 0.5 sp/bin.
- This is the opposite of what a threshold-masking artifact would produce.
  The FEM correction is real and its modest pooled magnitude reflects the
  dominance of low-rate pairs with near-zero correlations to correct.

**Open question:** Corrected noise correlations are strongly negative for
well-driven pairs (z ≈ -0.20 at 0.5 sp/bin). This could reflect: (a) genuine
anti-correlations after removing shared FEM drive, (b) overcorrection if the
rate covariance estimator captures more than FEM-driven variability, or
(c) the competitive surround in V1 producing negative noise correlations that
are normally masked by positive FEM-driven correlations.

## Remaining Controls (priority order)

1. ~~Firing rate inclusion threshold sweep~~ DONE (Panel S4)
2. ~~Isolation quality (good vs MUA) comparison~~ DONE (Panels S1-S3)
3. ~~r_SC vs geometric mean rate (before/after correction)~~ DONE (Panel S2c)
4. Stationarity: early/late split
5. Window-size convergence of 1-alpha
6. Sub-Poisson FF validation
7. Distance-dependent noise correlations
8. Signal-noise correlation structure
