#!/usr/bin/env python3
"""Write an exact methods-and-interpretation report for the true-history checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import OUT_DIR


CHECKPOINT_DIR = OUT_DIR / "preliminary_true_only"
TRUE_BANK = "real_trace_true_history_v1"
LEGACY_BANK = "legacy_wrapped_prefix"


def percent(row: dict[str, Any]) -> str:
    return (
        f"{float(row['ssi_percent_vs_stabilized']):.2f}% "
        f"(95% CI {float(row['ci95_low_paired_image_boot']):.2f} to "
        f"{float(row['ci95_high_paired_image_boot']):.2f})"
    )


def impact(value: dict[str, Any]) -> str:
    return (
        f"{float(value['point_percent_points']):.2f} pp "
        f"(95% CI {float(value['ci95_low_percent_points']):.2f} to "
        f"{float(value['ci95_high_percent_points']):.2f})"
    )


def main() -> int:
    stats = json.loads((CHECKPOINT_DIR / "statistics.json").read_text(encoding="utf-8"))
    if stats["status"] != "COMPLETE_TRUE_HISTORY_PRIMARY__HELD_CONTROL_PENDING":
        raise RuntimeError("The detailed primary report requires all 100 true-history images")
    curves = pd.read_csv(CHECKPOINT_DIR / "preliminary_true_only_curves.csv")
    summary = pd.read_csv(CHECKPOINT_DIR / "preliminary_true_only_summary.csv")
    effects = pd.read_csv(CHECKPOINT_DIR / "preliminary_true_only_unit_effects.csv")
    bank_manifest = json.loads((OUT_DIR / "banks/bank_manifest.json").read_text(encoding="utf-8"))
    core_manifest = json.loads(
        (
            OUT_DIR
            / "core_ssi/real_trace_true_history_v1/images_000_050/manifest.json"
        ).read_text(encoding="utf-8")
    )
    renderer = json.loads((OUT_DIR / "renderer_validation.json").read_text(encoding="utf-8"))
    previous_tf_stats_path = ROOT / "outputs/figures/fig4/spatiotemporal_tuning/analysis/statistics.json"
    previous_tf = json.loads(previous_tf_stats_path.read_text(encoding="utf-8"))

    def summary_row(bank: str, context: str, group: str) -> dict[str, Any]:
        return summary[
            summary["bank"].eq(bank)
            & summary["context"].eq(context)
            & summary["sf_group"].eq(group)
        ].iloc[0].to_dict()

    def curve_row(bank: str, context: str, group: str, bin_index: int) -> dict[str, Any]:
        return curves[
            curves["bank"].eq(bank)
            & curves["context"].eq(context)
            & curves["sf_group"].eq(group)
            & curves["bin_index"].eq(bin_index)
        ].iloc[0].to_dict()

    group_label = {
        "all_units": "all 100 units",
        "low_sf_lt0p5": "low SF (71 units)",
        "high_sf_ge0p5": "high SF (29 units)",
    }
    correction = stats["correction_impact_legacy_minus_true"]
    summary_lines = []
    for context in ("drift_only", "microsaccade"):
        for group in ("all_units", "low_sf_lt0p5", "high_sf_ge0p5"):
            summary_lines.append(
                "| "
                + " | ".join(
                    (
                        "drift only" if context == "drift_only" else "microsaccade-containing",
                        group_label[group],
                        percent(summary_row(LEGACY_BANK, context, group)),
                        percent(summary_row(TRUE_BANK, context, group)),
                        impact(correction[context][group]),
                    )
                )
                + " |"
            )

    def dose_table(context: str, n_bins: int) -> str:
        output = [
            "| Bin | Median path (arcmin) | Corrected low SF | Legacy low SF | Corrected high SF | Legacy high SF |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for bin_index in range(n_bins):
            tl = curve_row(TRUE_BANK, context, "low_sf_lt0p5", bin_index)
            ll = curve_row(LEGACY_BANK, context, "low_sf_lt0p5", bin_index)
            th = curve_row(TRUE_BANK, context, "high_sf_ge0p5", bin_index)
            lh = curve_row(LEGACY_BANK, context, "high_sf_ge0p5", bin_index)
            output.append(
                f"| {bin_index} | {float(tl['path_median_arcmin']):.2f} | "
                f"{float(tl['ssi_percent_vs_stabilized']):.2f}% | "
                f"{float(ll['ssi_percent_vs_stabilized']):.2f}% | "
                f"{float(th['ssi_percent_vs_stabilized']):.2f}% | "
                f"{float(lh['ssi_percent_vs_stabilized']):.2f}% |"
            )
        return "\n".join(output)

    drift_low_first = curve_row(TRUE_BANK, "drift_only", "low_sf_lt0p5", 0)
    drift_low_last = curve_row(TRUE_BANK, "drift_only", "low_sf_lt0p5", 7)
    drift_high_first = curve_row(TRUE_BANK, "drift_only", "high_sf_ge0p5", 0)
    drift_high_peak = curve_row(
        TRUE_BANK,
        "drift_only",
        "high_sf_ge0p5",
        int(stats["true_history"]["high_peak_bin_index"]),
    )
    drift_high_last = curve_row(TRUE_BANK, "drift_only", "high_sf_ge0p5", 7)
    micro_low_first = curve_row(TRUE_BANK, "microsaccade", "low_sf_lt0p5", 0)
    micro_low_last = curve_row(TRUE_BANK, "microsaccade", "low_sf_lt0p5", 4)
    micro_high_first = curve_row(TRUE_BANK, "microsaccade", "high_sf_ge0p5", 0)
    micro_high_last = curve_row(TRUE_BANK, "microsaccade", "high_sf_ge0p5", 4)

    low_effect = effects[effects["sf_group"].eq("low_sf_lt0p5")][
        "legacy_minus_true_benefit_percent_points"
    ]
    high_effect = effects[effects["sf_group"].eq("high_sf_ge0p5")][
        "legacy_minus_true_benefit_percent_points"
    ]
    model = core_manifest["model"]
    model_core = model["model"]
    checkpoint = model_core["checkpoint_path"]
    rr_version = model["rr100_version"]
    low_contrast = stats["true_history"]["low_last_minus_first_absolute_ssi"]
    high_contrast = stats["true_history"]["high_peak_minus_last_absolute_ssi"]

    report = f"""# Figure 4 causal-history correction: exact methods, results, and interpretation

## Read this first

This document reports **one completed analysis stage**: correction of the noncausal temporal prefix in the core Figure 4 natural-image/FEM computation, followed by a full rescore of the primary real-history bank.

It is **not** the requested final mechanism audit. In particular, this stage did not run a new temporal-frequency analysis, q analysis, learned-frontend analysis, retinal space-to-time spectrum, causal event-history split, step-and-hold experiment, layerwise hook analysis, or frontend intervention. Those analyses were deliberately not started after we shortened the computation for a same-day result. The held-prefix trajectories were constructed and validated, but their model responses were not scored.

The correct one-sentence interpretation is:

> Replacing the legacy noncausal prefix with the real recorded pre-snippet eye history modestly reduces Figure 4 effect sizes but leaves the central low-/high-SF movement-scale interaction intact in the complete primary bank.

This supports robustness of the core phenomenon to the history correction. It does **not** yet explain the phenomenon mechanistically.

## 1. What was wrong in the legacy computation

The model consumes 32 retinal frames per output. Running the exact production lag embedder on a synthetic sequence `e[t] = t` established that lag 0 is the current frame and lag 31 is the oldest frame. A model output at scored time `t` therefore must see the chronological source interval:

```text
e[t-31], e[t-30], ..., e[t]
```

Thus, a 40-output snippet requires 31 genuinely preceding samples plus the 40 scored samples: 71 source frames in total.

The legacy helper instead supplied:

```text
e[0], e[1], ..., e[31], e[0], e[1], ..., e[39]
```

It generated 41 lagged outputs and discarded the first. Of the 40 retained outputs, outputs 0–30 contained future samples relative to their nominal output time and the artificial `e[31] → e[0]` wrap. Only outputs 31–39 had ordinary causal sliding histories. This is why the frozen old bank is called `{LEGACY_BANK}` here.

## 2. Exactly how the corrected trajectory bank was built

For each of the same 1,000 stored Figure 4 trajectories, I used its saved session, trial/fixation, source-window boundaries, and scored start index to retrieve samples `start-31 ... start+39` from the original 120-Hz BackImage eye-position recording.

- No downsampling was applied.
- No filtering was applied.
- Every recovered sample had to be finite and remain inside the selected source fixation window.
- The historical builder had centered the scored 40 samples by subtracting their two-dimensional mean. I calculated that mean once from the scored source samples and subtracted the same translation from the complete 71-frame segment. I did not center the prefix separately.
- The transformed scored samples reproduced the frozen 40-sample bank with maximum error {float(bank_manifest['max_reconstruction_error']):g} degrees.
- All 1,000 trajectories had sufficient valid history; none was excluded.

The resulting primary bank, `{TRUE_BANK}`, contains real source samples for times `-31 ... 39`. A second bank, `real_trace_held_initial_history_v1`, was also constructed by holding all 31 prefix samples at `e[0]` and then appending the real scored trajectory. That control bank was validated but **not passed through the model in this shortened run**.

Across each bank's 40,000 trajectory-output histories, validation found:

- 0 outputs with a future source sample;
- 0 outputs with nonmonotonic source indices;
- 0 artificial prefix/scored boundary discontinuities;
- 0 trajectories with incomplete history;
- 0 excluded trajectories.

The largest true-history boundary-step discrepancy from the raw source trace was {float(bank_manifest['max_true_boundary_step_error_vs_source_deg']):.3g} degrees (float32 precision). On an actual image/trajectory render, the corrected last nine outputs—which overlap the valid portion of the legacy construction—matched the legacy pixels with maximum absolute error {float(renderer['corrected_vs_legacy_valid_overlap_max_abs_pixel_error']):g}.

## 3. Model, stimuli, units, and rescore

No model was retrained and no weight was changed.

- Checkpoint: `{checkpoint}`
- Checkpoint SHA-256: `{model_core['checkpoint_sha256']}`
- Model configuration: `{model_core['hparams']['model_cfg']}`
- Dataset configuration: `{model_core['dataset_configs']}`
- Representative population: the exact 100 Figure 4 RR medoid channels, version `{rr_version}`
- Canonical model readout: {int(model['canonical_readout_n_units'])} channels; the scorer took an algebraically exact one-hot slice for the 100 selected channels. An equivalence check gave zero readout error.
- Model input: 151×151 pixels at {float(model['stimulus']['ppd']):.8f} pixels/degree.
- Sampling: 120 Hz, 32-frame causal input per output.
- Each stored trajectory: 40 scored samples (sample times span 325 ms; 40 rate bins total 333 ms).
- Natural images: the same 100 Figure 4 image patches.
- Crossed evaluation: every image with every trajectory, for 100,000 movies and 4,000,000 scored output times, retaining all 100 units.

For every image/trajectory/unit, the saved primary outputs are SSI, expected spikes, mean rate, and population SSI. All merged arrays have their exact expected dimensions and contain only finite values; their SHA-256 hashes match the merge manifest.

## 4. Exactly how SSI and the plotted percentage were calculated

For each model rate map `r(x,y)` at one time point, the spatial mean was `r̄` and the normalized gain map was `g(x,y)=r(x,y)/r̄`. The per-frame spatial selectivity quantity was:

```text
SSI_t = mean_xy [ g(x,y) log2(g(x,y)) ]
```

The 40 frames were then combined using their expected spike counts `r̄_t Δt` as weights:

```text
SSI_movie,unit = Σ_t SSI_t r̄_t Δt / Σ_t r̄_t Δt
```

When units and trajectories were pooled for a curve, the same expected-spike numerator and denominator were summed, rather than taking an unweighted mean of millions of rows. The vertical quantity in the figure is:

```text
100 × (SSI_motion − SSI_stabilized) / SSI_stabilized
```

The stabilized reference is the frozen per-image Figure 4 baseline. Therefore, a value of 20% means the moving condition's spike-weighted SSI is 20% larger than its stabilized reference; it does not mean 20 percentage points of raw SSI and it is not a firing-rate change.

## 5. Groups, trajectory bins, and uncertainty

Nothing was regrouped after seeing the correction.

- Low SF: historical `sf_split_metric < 0.5` cpd, 71 units.
- High SF: historical `sf_split_metric ≥ 0.5` cpd, 29 units.
- Drift only: 800 trajectories with zero cached microsaccade events in the 40-sample scored interval.
- Microsaccade-containing: 200 trajectories with at least one cached event in the scored interval.
- Drift trajectories were sorted by the frozen rendered path length and divided into eight equal-count bins of 100.
- Microsaccade trajectories were divided into five equal-count bins of 40.

The error bars are 95% paired **image-bootstrap** intervals from 10,000 resamples, using seed family 47 and the frozen Figure 4 ratio-difference convention. Images are the resampling unit. These intervals quantify generalization across the 100 images, not across new model units or new trajectories.

## 6. Numerical results

| Scored-interval class | Population | Legacy wrapped prefix | Corrected true history | Legacy − corrected |
|---|---|---:|---:|---:|
{chr(10).join(summary_lines)}

For the primary drift-only result:

- Overall corrected benefit: {percent(stats['true_history']['overall_drift'])}.
- Low-SF corrected benefit: {percent(stats['true_history']['low_sf_drift'])}.
- High-SF corrected benefit: {percent(stats['true_history']['high_sf_drift'])}.
- The low-SF curve had Spearman `rho={float(stats['true_history']['low_curve_spearman_rho']):.3f}` across its eight path bins. Its last-minus-first absolute-SSI contrast was {float(low_contrast['point']):.5f}, with CI [{float(low_contrast['ci95_low']):.5f}, {float(low_contrast['ci95_high']):.5f}].
- The high-SF maximum occurred at bin {int(stats['true_history']['high_peak_bin_index'])}, whose median path was {float(drift_high_peak['path_median_arcmin']):.2f} arcmin. Peak-minus-last absolute SSI was {float(high_contrast['point']):.5f}, with CI [{float(high_contrast['ci95_low']):.5f}, {float(high_contrast['ci95_high']):.5f}].

The per-unit drift summary is heterogeneous and is not the estimator used for the population curves. Across the 71 low-SF units, legacy-minus-corrected benefit had mean {float(low_effect.mean()):.2f} pp and median {float(low_effect.median()):.2f} pp. Across the 29 high-SF units, it had mean {float(high_effect.mean()):.2f} pp and median {float(high_effect.median()):.2f} pp.

### Drift-only plotting values

{dose_table('drift_only', 8)}

### Microsaccade-containing plotting values

{dose_table('microsaccade', 5)}

## 7. How to read `fig_true_history_primary`

The x-axis in all panels is the median 40-sample path length of one equal-count trajectory bin. The y-axis is the percent SSI change from stabilization defined above. Teal circles are the corrected real-history responses; red crosses are the frozen legacy wrapped-prefix responses. Error bars are paired-image bootstrap intervals.

### Panel A — drift-only, low SF

The corrected low-SF curve rises from {float(drift_low_first['ssi_percent_vs_stabilized']):.2f}% at a median path of {float(drift_low_first['path_median_arcmin']):.2f} arcmin to {float(drift_low_last['ssi_percent_vs_stabilized']):.2f}% at {float(drift_low_last['path_median_arcmin']):.2f} arcmin. Every successive point is higher, yielding `rho=1`. The red curve is generally slightly above the teal curve, so the bug inflated the magnitude, but it did not create the positive dose relationship.

### Panel B — drift-only, high SF

The corrected high-SF curve begins at {float(drift_high_first['ssi_percent_vs_stabilized']):.2f}%, rises shallowly to {float(drift_high_peak['ssi_percent_vs_stabilized']):.2f}% near {float(drift_high_peak['path_median_arcmin']):.2f} arcmin, then falls to {float(drift_high_last['ssi_percent_vs_stabilized']):.2f}% in the largest-path bin. The positive peak-minus-last CI shows that this downturn is not just the ordering of noisy point estimates under the frozen image-bootstrap analysis. The red and teal curves are close throughout.

Panels A and B together are the central Figure 4 interaction: low-SF units continue to benefit as movement grows, while high-SF units have a smaller useful movement range and decline at the largest paths. That interaction survives real causal history.

### Panel C — microsaccade-containing, low SF

The corrected curve increases from {float(micro_low_first['ssi_percent_vs_stabilized']):.2f}% to {float(micro_low_last['ssi_percent_vs_stabilized']):.2f}%. Here the legacy red curve is noticeably higher: the overall low-SF microsaccade correction is {impact(correction['microsaccade']['low_sf_lt0p5'])}. Thus, the bad prefix mattered more for this subset, but the corrected curve still shows a large increasing benefit.

### Panel D — microsaccade-containing, high SF

The corrected curve decreases from {float(micro_high_first['ssi_percent_vs_stabilized']):.2f}% to {float(micro_high_last['ssi_percent_vs_stabilized']):.2f}% across the five bins. Legacy and corrected values remain relatively close; the aggregate correction is {impact(correction['microsaccade']['high_sf_ge0p5'])}.

The microsaccade label used here describes only the scored 40-sample interval. This analysis did not yet reclassify each output by fast events in its complete true causal history, so Panels C/D do not answer whether a preceding event drives the response.

## 8. What this result does and does not establish

### Supported by this completed stage

- FEM motion still increases spike-weighted SSI relative to stabilization after replacing the noncausal prefix with real recorded history.
- The low-SF drift curve remains strongly progressive with path length.
- The high-SF drift curve retains a statistically supported intermediate maximum and decline at the largest paths.
- The legacy prefix modestly inflated the all-unit drift estimate ({impact(correction['drift_only']['all_units'])}) and had a larger effect on low-SF microsaccade-containing trials.
- The central qualitative Figure 4 interaction is robust in the primary true-history bank.

### Not established by this stage

- The final correction-gate label `CORE RESULT SURVIVES`: the response-level held-prefix control was deferred.
- Whether real preceding motion matters relative to a no-prior-motion held prefix.
- Whether the “drift-only” scored subset contains a recent fast event in the preceding 31-frame history.
- Any explanation in terms of temporal frequency, `v*=ft*/fs*`, q, or temporal-band matching.
- Whether motion-created temporal power is transmitted by particular learned frontend filters.
- Whether the effect arises first in the frontend, stem, residual blocks, ConvGRU, or readout.
- Whether a step displacement, sustained drift, and a real saccade have separable effects.

The label **{stats['provisional_signature']}** means only that the complete primary bank passes the same three qualitative predicates used by the planned gate: positive overall drift CI, progressive low-SF curve, and significant interior high-SF peak. It is not a substitute for the unrun held-prefix comparison.

## 9. Where temporal frequency stands

There is no new temporal-frequency panel in `fig_true_history_primary` because that figure answers a prior question: **did the core Figure 4 relationship survive causal-history correction?** It does not ask why it survives.

A separate earlier grating probe did measure the 100 units on six spatial frequencies, four orientations, and a dense 0.2–51.2-Hz temporal-frequency grid. That probe found {int(previous_tf['n_interior_tf_preference'])} interior TF maxima and {int(previous_tf['n_boundary_tf_preference'])} censored/boundary maxima ({int(previous_tf['tf_censoring_counts']['left'])} left, {int(previous_tf['tf_censoring_counts']['right'])} right). Using the fitted spatial-frequency estimates gave median `v*=ft*/fs*` values of {float(previous_tf['median_predicted_speed_low_deg_s']):.2f} deg/s for low-SF units and {float(previous_tf['median_predicted_speed_high_deg_s']):.3f} deg/s for high-SF units.

Those numbers did **not** yield the hoped-for temporal-band collapse. Expressing the old observational curves in q coordinates reduced the low/high RMS separation by only {100.0 * (1.0 - float(previous_tf['q_to_raw_curve_rms_ratio'])):.1f}%, and the old controlled unit-level speed correlation was `rho={float(previous_tf['controlled_scaling']['speed_correlation']['rho']):.3f}` with CI [{float(previous_tf['controlled_scaling']['speed_correlation']['ci95_low']):.3f}, {float(previous_tf['controlled_scaling']['speed_correlation']['ci95_high']):.3f}]. The earlier conclusion was therefore only “Partial.”

However, that earlier q/SSI association and controlled scaling used the now-invalid wrapped-prefix natural-image scorer. The grating tuning tensor itself is a separate model probe and is not invalidated by the FEM prefix bug, but its connection to corrected natural-image SSI has not been rerun. In addition, the later mechanism specification required audits that remain unfinished: physical identifiability of the very low SF values, q support, output time courses after transient removal, high-TF sampling artifacts, and the learned frontend's actual frequency response. Consequently, the current evidence does **not** support a manuscript claim that output-unit temporal-band matching explains Figure 4.

Existing temporal-probe details are in `outputs/figures/fig4/spatiotemporal_tuning/analysis/report.md`; they should be read as pre-correction diagnostics, not as the final mechanism result.

## 10. What remains to answer the original mechanism question

The analyses discussed but not performed in this shortened stage are:

1. Score the held-prefix response bank and compare true versus held history.
2. Classify every corrected output by fast events in its actual 31-frame source history.
3. Generate retinal x–t slices and spatiotemporal power spectra across controlled motion scales.
4. Extract every learned temporal frontend kernel and calculate its Fourier response.
5. Run corrected step-and-hold, constant-velocity, and saccade-plus-drift experiments.
6. Hook raw input, frontend, stem, residual blocks, ConvGRU, and readout to localize where spatial concentration changes.
7. Run matched frontend interventions.
8. Audit SF aperture identifiability and TF time courses; only then revisit q on the corrected outputs.

Until those are done, the defensible conclusion is **robust phenomenon, mechanism unresolved**.

## 11. Reproducible outputs

- Detailed report: `TRUE_HISTORY_PRIMARY_REPORT.md`
- Primary figure: `fig_true_history_primary.pdf`, `.svg`, `.png`
- Exact figure data: `plot_data/fig_preliminary_true_only.csv`
- Aggregate table: `preliminary_true_only_summary.csv`
- Per-bin table: `preliminary_true_only_curves.csv`
- Per-unit table: `preliminary_true_only_unit_effects.csv`
- Machine-readable results: `statistics.json`, `preliminary_true_only_outputs.npz`
- Verification: `verification.json`
- Causal audit: `../CAUSAL_INDEX_AUDIT.md`, `../causal_index_mapping.csv`
- Source-history provenance: `../true_history_trajectory_table.csv`
- Validation: `../history_validation.csv`, `../renderer_validation.json`
- Merged primary model outputs: `../core_ssi/real_trace_true_history_v1/merged/`
"""
    (CHECKPOINT_DIR / "TRUE_HISTORY_PRIMARY_REPORT.md").write_text(report, encoding="utf-8")
    print(CHECKPOINT_DIR / "TRUE_HISTORY_PRIMARY_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
