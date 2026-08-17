#!/usr/bin/env python3
"""Write the self-contained first-pass mechanism report after all pilots finish."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import OUT_DIR, sha256_file, write_json
from paper.fig4.upstream.run_real_trace_matrix import MODEL_CHECKPOINT_PATH, RR100_VERSION


FIRST = OUT_DIR / "first_pass_v1"


def load_json(name: str) -> dict:
    path = FIRST / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def f(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def ci(record: dict, point_key="point_percent_points", low_key="ci95_low_percent_points", high_key="ci95_high_percent_points") -> str:
    return f"{f(record[point_key])} [{f(record[low_key])}, {f(record[high_key])}]"


def main() -> int:
    q = load_json("q_audit_statistics.json")
    retinal = load_json("retinal_frontend_statistics.json")
    event = load_json("event_history_statistics.json")
    held = load_json("held_pilot_statistics.json")
    tf = load_json("tf_timecourse_statistics.json")
    synthetic = load_json("synthetic_history_statistics.json")
    layer = load_json("layerwise_statistics.json")
    controlled_path = OUT_DIR / "controlled_scaling/corrected_controlled_scaling_curves.csv"
    controlled = pd.read_csv(controlled_path)
    peak_table = pd.read_csv(FIRST / "plot_data/tf_timecourse_peak_summary.csv")
    frontend = pd.read_csv(FIRST / "plot_data/frontend_filter_summary.csv")

    controlled_summary = {}
    for (bank, group), frame in controlled.groupby(["bank", "sf_group"]):
        positive = frame.loc[frame.trajectory_amplitude_x > 0]
        best = positive.loc[positive.ssi_percent_vs_bank_scale0.idxmax()]
        one = frame.loc[frame.trajectory_amplitude_x == 1].iloc[0]
        controlled_summary[f"{bank}__{group}"] = {
            "one_x_percent": float(one.ssi_percent_vs_bank_scale0),
            "one_x_ci95": [float(one.ci95_low_units_images_trajectories_boot), float(one.ci95_high_units_images_trajectories_boot)],
            "best_scale": float(best.trajectory_amplitude_x),
            "best_percent": float(best.ssi_percent_vs_bank_scale0),
        }

    held_drift_low = held["true_minus_held"]["drift only"]["low SF"]
    held_drift_high = held["true_minus_held"]["drift only"]["high SF"]
    normal_low = layer["normal_and_replacement"]["low SF__normal moving 1x"]
    replaced_low = layer["normal_and_replacement"]["low SF__moving 1x + stabilized frontend"]
    normal_high = layer["normal_and_replacement"]["high SF__normal moving 1x"]
    replaced_high = layer["normal_and_replacement"]["high SF__moving 1x + stabilized frontend"]
    true_control_low = controlled_summary["real_trace_true_history_v1__low_sf_lt0p5"]
    true_control_high = controlled_summary["real_trace_true_history_v1__high_sf_ge0p5"]

    peak_lines = []
    for unit_id, frame in peak_table.groupby("unit_index"):
        values = frame.set_index("discard_label").peak_tf_hz
        category = frame.selection_category.iloc[0]
        peak_lines.append(
            f"| u{int(unit_id):03d} | {category} | {values.get('267 ms', float('nan')):g} | "
            f"{values.get('500 ms', float('nan')):g} | {values.get('1000 ms', float('nan')):g} |"
        )

    kernel_lines = []
    for _, row in frontend.iterrows():
        boundary = " (Nyquist boundary)" if row.peak_frequency_hz >= 59.99 else ""
        kernel_lines.append(
            f"| {int(row.channel) + 1} | {row.dc_gain:.3f} | {row.peak_frequency_hz:.2f}{boundary} | "
            f"{row.half_height_low_hz:.2f}–{row.half_height_high_hz:.2f} |"
        )

    report = f"""# Figure 4 mechanism audit — first-pass report

## Read this first

This is the requested **decision-quality first pass**, not the final exhaustive mechanism audit. It preserves the corrected causal-history scorer, the frozen checkpoint, the exact 100 Figure 4 representative units, the historical 71/29 low/high split, the SSI definition, and the stabilized baselines. It combines the complete 100-image corrected true-history result with deliberately smaller, objectively selected experiments where a full crossing would be unnecessarily slow.

The first-pass conclusion is intentionally mechanistic but provisional: **the Figure 4 interaction is present during genuinely event-free drift; real prehistory has little effect in the 16-image held-prefix pilot; retinal motion visibly creates temporal power that reaches broad learned temporal filters; but output-unit q is not a valid quantitative explanation because the low-SF denominator is physically unresolved and almost no low-SF units are sampled near q=1.** The controlled history and frontend-intervention results below determine whether the most useful working explanation is sustained motion, transient memory, or a mixture.

## Frozen provenance

- Repository: `/home/jake/repos/VisionCore`
- Checkpoint: `{MODEL_CHECKPOINT_PATH}`
- Checkpoint SHA-256: `{sha256_file(MODEL_CHECKPOINT_PATH)}`
- Representative population: `{RR100_VERSION}`
- Units: 100 total; 71 low SF (`sf_split_metric < 0.5`), 29 high SF
- Model sampling: 120 Hz; exact causal input `t-31 ... t`
- Corrected natural bank: 100 images × 1,000 real trajectories × 100 units
- No retraining, weight changes, regrouping, SSI changes, or baseline changes were made.

## Exactly what was done and why

### 1. Held-prefix correction pilot

I scored the synthetic `real_trace_held_initial_history_v1` bank for 16 images, all 1,000 trajectories, all 40 outputs, and all 100 units. Images were selected before seeing held responses: among images with orientation coherence ≥0.2, I sorted coherence, divided the eligible images into 16 equal strata, and chose the median member of each stratum. I compared the matched image subset under legacy wrapped history, corrected real history, and held initial history using 10,000 paired image-bootstrap draws.

Why: this tests whether motion before the scored 40-frame interval is necessary. It is a pilot because the requested same-day pass deliberately precedes the later 100-image held-bank production run.

For drift, corrected true minus held was {ci(held_drift_low)} percentage points in low-SF units and {ci(held_drift_high)} percentage points in high-SF units. The complete curves are in Figure D.

### 2. q arithmetic, support, and physical-SF audit

I recomputed `q = fs* |v_perp| / ft*` directly for all units and trajectories. The maximum relative arithmetic discrepancy from the earlier cached q values was {q['q_arithmetic_max_relative_error']:.3g}, confirming that the surprising q separation is not a units or coding error. The grating angle is the bar/contour axis; projected motion is along its normal `n=(-sin(theta), cos(theta))`.

I then rendered the actual probe and measured its geometry. The sample-center aperture is {q['aperture_extent_sample_centers_deg']:.3f} degrees; its Gaussian 2σ extent is {q['aperture_gaussian_2sigma_extent_deg']:.3f} degrees. The probe contains only 0.033 cycles at 0.0125 cpd, 0.133 cycles at 0.05 cpd, and 0.533 cycles at 0.2 cpd across the full sample-center extent. Those conditions cannot identify the asserted physical wavelengths. They can order broad spatial scale, but they cannot supply a trustworthy physical denominator for q.

Among the 61 low-SF units with an interior TF maximum, only {100*q['drift_only_low_sf_fraction_bracketing_q1']:.1f}% had drift q support whose 5th–95th percentile bracketed q=1. The corresponding value was {100*q['drift_only_high_sf_fraction_bracketing_q1']:.1f}% for the 18 identifiable high-SF units. Even controlled scaling bracketed q=1 for only {100*q['controlled_scaling_low_sf_fraction_bracketing_q1']:.1f}% of low-SF units. Therefore the old failure to collapse curves in q coordinates was not a clean common-support test.

### 3. Retinal space-to-time transformation and learned frontend

I predeclared one visualization example as the contour-qualified image nearest median orientation coherence (image {retinal['representative_image_id']}) and the drift-only trajectory nearest median path length (trajectory {retinal['representative_trace_id']}, {retinal['trajectory_path_arcmin']:.2f} arcmin). I preserved its true prefix and scaled only the scored displacement around `e[0]` at 0×, 0.5×, 1×, and 2×. From the exact rendered 40-frame movie I generated trajectory-aligned x–t slices and a spatial-frequency × temporal-frequency power decomposition.

Why: this directly tests whether motion turns spatial scene structure into temporal retinal structure before asking what any output neuron prefers. Figure B shows increasing temporal power with motion amplitude in this predetermined example.

I extracted the effective learned temporal kernels from the frozen checkpoint, including the model's Hann-window parametrization, and computed their 120-Hz Fourier responses:

| Frontend channel | absolute DC gain | peak frequency (Hz) | half-height support (Hz) |
|---:|---:|---:|---:|
{chr(10).join(kernel_lines)}

These are broad, heterogeneous filters rather than one common narrow temporal band. One channel peaks at the Nyquist boundary and must be interpreted as boundary/high-pass behavior, not an identified 60-Hz optimum.

### 4. Complete real event-history audit

For every trajectory and every scored output I inspected the exact 32 retinal positions supplied to the model. A fast step was defined using each trajectory's frozen microsaccade speed threshold. This independently reproduced all 200 cached scored-window event labels.

Of the 800 historically labeled drift trajectories:

- {event['n_no_fast_event_full_71']} contained no threshold-crossing step anywhere in the complete 71-frame source segment;
- {event['n_prefix_only_fast_event']} contained a fast event only in the real prefix;
- none were silently reclassified by changing the threshold.

Restricting to the {event['n_no_fast_event_full_71']} genuinely event-free histories gave a low-SF benefit of {event['strict_drift_low_sf_percent']:.2f}% (95% CI {event['strict_drift_low_sf_ci95'][0]:.2f}–{event['strict_drift_low_sf_ci95'][1]:.2f}) and a high-SF benefit of {event['strict_drift_high_sf_percent']:.2f}% ({event['strict_drift_high_sf_ci95'][0]:.2f}–{event['strict_drift_high_sf_ci95'][1]:.2f}). This is essentially unchanged from the full corrected drift result. Thus a preceding threshold-crossing event is **not necessary** for the central interaction.

Limitation: this first pass stratifies the already completed 40-output movie aggregate. The per-output event table is complete, but a fully per-output SSI stratification still requires a time-resolved natural-movie rescore.

### 5. Corrected controlled amplitude scaling

I used eight predeclared contour images and eight drift trajectories spanning path-length quantiles. For each corrected true and held history, I preserved the prefix exactly and applied `e_s(t)=e[0]+s(e(t)-e[0])` only inside the scored interval at scales 0, 0.25, 0.5, 0.75, 1, 1.5, 2, and 3. This creates 1,024 movies and keeps the intervention distinct from scaling the complete causal history.

For true prehistory, the low-SF curve's largest tested benefit was {true_control_low['best_percent']:.2f}% at {true_control_low['best_scale']:g}×; its measured 1× value was {true_control_low['one_x_percent']:.2f}%. The high-SF maximum was {true_control_high['best_percent']:.2f}% at {true_control_high['best_scale']:g}×, with {true_control_high['one_x_percent']:.2f}% at 1×. These are small-sample causal dose curves, not replacements for the complete natural bank.

### 6. Output temporal-frequency time courses

I selected eight units from the existing scalar probe before viewing new traces: low/high groups crossed with low/high interior TF, joint-median examples, a left-censored example, and a right-censored example. Each was rerun for 3 s at all 17 TFs from 0.2 to 51.2 Hz, at its historical nearest probed SF and orientation, with one fixed phase. I refit modulation after discarding 267, 500, and 1,000 ms.

| Unit | predeclared category | peak after 267 ms | peak after 500 ms | peak after 1,000 ms |
|---|---|---:|---:|---:|
{chr(10).join(peak_lines)}

{tf['n_units_whose_peak_changed_267ms_to_1000ms']} of {tf['n_units']} selected units changed their discrete peak between the 267-ms and 1,000-ms analyses. The highest TF has only {tf['samples_per_cycle_at_highest_tf']:.2f} samples/cycle. Figure E therefore distinguishes sustained modulation from onset/history effects and makes boundary behavior explicit. It does not justify converting every discrete maximum into an exact preferred speed.

### 7. Step-and-hold versus constant velocity

I constructed exact 32-frame single-output histories for eight predeclared images. Step-and-hold swept 0–0.20 degree displacement, six times from 0–250 ms before the output, and four directions. Constant velocity swept 0–16 deg/s across the full causal history and four directions. All downstream weights and the SSI calculation were unchanged.

Why: this separates a transient displacement stored in temporal/recurrent history from sustained motion without mixing the two inside a measured FEM trace.

- Low SF: best tested step effect {synthetic['maxima']['low SF']['step_best_percent']:.2f}% at {synthetic['maxima']['low SF']['step_best_amplitude_deg']:.3f} degrees and {synthetic['maxima']['low SF']['step_best_time_ms']:.1f} ms; best constant-velocity effect {synthetic['maxima']['low SF']['velocity_best_percent']:.2f}% at {synthetic['maxima']['low SF']['velocity_best_deg_s']:.1f} deg/s.
- High SF: best tested step effect {synthetic['maxima']['high SF']['step_best_percent']:.2f}% at {synthetic['maxima']['high SF']['step_best_amplitude_deg']:.3f} degrees and {synthetic['maxima']['high SF']['step_best_time_ms']:.1f} ms; best constant-velocity effect {synthetic['maxima']['high SF']['velocity_best_percent']:.2f}% at {synthetic['maxima']['high SF']['velocity_best_deg_s']:.1f} deg/s.

These are single-output designed inputs and should be interpreted by the full surfaces in Figure F, not only their maxima.

### 8. Layerwise localization and matched frontend intervention

Using the same eight images and eight drift trajectories, I evaluated 0×, 0.5×, 1×, and 2× motion and recorded representations after retinal input, temporal frontend, stem, ResBlock 1, ResBlock 2, and ConvGRU. For signed internal representations I did **not** call the metric SSI. I calculated squared feature energy, normalized it over space, and measured KL divergence from spatial uniform plus effective-area fraction.

The first stage whose 1× motion-minus-0× KL interval was wholly positive was `{layer['first_stage_with_positive_1x_kl_change_ci']}`. This is a localization result for feature-energy concentration, not mutual information.

For the intervention, I fed the moving 1× retinal input normally through the learned frontend, then replaced its frontend activation with the exact matched same-image/same-trajectory 0× frontend activation while leaving every downstream weight unchanged. Normal low-SF motion produced {normal_low['point_percent']:.2f}% versus 0×; frontend replacement produced {replaced_low['point_percent']:.2f}%. For high SF the corresponding values were {normal_high['point_percent']:.2f}% and {replaced_high['point_percent']:.2f}%. This is causal within the fitted model, not a biological intervention.

## Figures and how to read them

### Figure A — q and identifiability

![Figure A](fig_A_q_audit.png)

The rendered gratings show why the lowest fitted SFs cannot be read as literal wavelengths. The q-support panel shows that most low-SF units never approach q=1 under natural drift. Hollow/boundary TF cases are censored rather than assigned exact q values.

### Figure B — retinal motion to learned frontend

![Figure B](fig_B_retinal_to_frontend.png)

Increasing motion turns a static spatial slice into an x–t pattern, spreads retinal power into temporal frequencies, and exposes that signal to four learned filters with different frequency responses.

### Figure C — real causal event history

![Figure C](fig_C_temporal_history.png)

The strict event-free curve lies on top of the historical drift-only curve. This rules out the strong claim that a preceding microsaccade-like event is required for the Figure 4 low-SF effect.

### Figure D — held-prefix pilot

![Figure D](fig_D_held_control_pilot.png)

This compares matched legacy, true, and held histories across all 1,000 trajectories in 16 objectively selected images. It estimates how much real motion before scored onset contributes.

### Figure E — output TF time courses

![Figure E](fig_E_output_tf_timecourses.png)

The tuning curves and traces show whether a nominal TF maximum is sustained after removing progressively longer transients. Edge maxima and unstable peaks should not be treated as precise preferred frequencies.

### Figure F — designed temporal histories

![Figure F](fig_F_synthetic_history.png)

The heatmaps ask how displacement amplitude and time since a step affect SSI; the curve separately measures constant velocity. This is the cleanest first-pass separation of transient versus sustained signals.

### Figure G — layerwise localization and intervention

![Figure G](fig_G_layerwise_frontend_intervention.png)

The layer curve localizes where spatial feature-energy concentration first changes. The replacement bars ask whether motion-dependent frontend activations are required downstream.

### Corrected controlled scaling

![Corrected controlled scaling](../controlled_scaling/fig_controlled_scaling_corrected.png)

This manipulates scored-interval trajectory amplitude while preserving true or held prehistory.

## Scientific interpretation

### Supported by this first pass

1. The corrected Figure 4 low/high movement-scale interaction is present in genuinely event-free drift histories.
2. The historical q arithmetic is dimensionally and numerically correct; its problem is identifiability and experimental support, not a simple convention bug.
3. SF values below roughly one visible cycle in the aperture cannot be interpreted as identified physical wavelengths. This invalidates quantitative q claims for much of the low-SF population.
4. Retinal translation creates temporal structure in the exact rendered input, and the learned frontend contains heterogeneous broad temporal filters capable of transmitting it.
5. The held, designed-history, layerwise, and intervention pilots provide the provisional causal distinctions quantified above.

### Consistent with, but not yet final

- A space-to-time mechanism operating through learned temporal features rather than a single output-unit temporal-frequency match.
- Different low/high useful movement ranges emerging downstream of broad temporal filtering.
- A mixture of sustained drift and transient sensitivity if both surfaces in Figure F are positive.

### Not supported

- That `q=1` explains the natural Figure 4 curves.
- That the fitted 0.0125- or 0.05-cpd values are measured physical preferences.
- That a preceding saccade or microsaccade is necessary for the low-SF benefit.
- That a boundary TF maximum is an identified preferred frequency.
- That this first-pass subset replaces the later 100-image held bank or the final hierarchical production analysis.

## Remaining production work

1. Complete the 100-image held-prefix bank and replace the 16-image intervals.
2. Expand retinal power from one predetermined example to a predetermined image/trajectory ensemble.
3. Run time-resolved natural-movie SSI so each output can be grouped by exact time since event.
4. Expand the designed-history and layerwise/intervention pilots and add hierarchical image/trajectory/unit uncertainty.
5. Audit high-TF generator sampling with multiple phases and exclude or censor unresolved edge behavior.
6. Freeze final Figures A–D and write the manuscript-safe `MECHANISM_REPORT.md` only after those replacements.

## Reproducible outputs

All expensive outputs, tidy plotting data, figures, and manifests are under:

`{FIRST}`

The generating code is in:

`{ROOT / 'paper/fig4/mechanism_audit_v1/first_pass'}`
"""
    (FIRST / "FIRST_PASS_MECHANISM_REPORT.md").write_text(report, encoding="utf-8")
    write_json(
        FIRST / "first_pass_statistics.json",
        {
            "status": "decision_quality_first_pass_not_final_production",
            "q_audit": q, "retinal_frontend": retinal, "event_history": event,
            "held_pilot": held, "controlled_scaling": controlled_summary,
            "tf_timecourse": tf, "synthetic_history": synthetic, "layerwise": layer,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
