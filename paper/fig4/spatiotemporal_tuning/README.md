# Figure 4 joint spatiotemporal-tuning test

This directory contains an additive analysis of the fixed Figure 4 model and
cached image × FEM bank. It does not modify the model, RR100 population, SSI
definition, source bank, stabilized baselines, or existing Figure 4 outputs.

## Native selected-twin path

The original workflow below records the historical 120-Hz analysis. A
genuinely native-240-Hz selected twin must instead use its own scored matrix,
cycle-valid tuning surface, and robust tuning summary. In particular,
The production retinal-spectrum analysis is
`analyze_image_specific_joint_engagement.py`. It measures image Fourier power
from the exact stabilized crops and computes the temporal spectrum of every
translated Fourier carrier from the complete native eye-position trajectory.
It cannot silently reuse the old M66 image-power cache or trace bank.

```bash
python paper/fig4/spatiotemporal_tuning/analyze_image_specific_joint_engagement.py \
  --matrix-dir /path/to/native/merged \
  --out-dir /path/to/native/image_specific_joint_engagement_phase_spectrum
```

For each image mode, the calculation uses
`R_k(t) = I_k exp(-i 2pi k.X(t))` and estimates the two-sided, mean-removed
trajectory-phase spectrum with two DPSS tapers. Positive and negative temporal
frequencies are folded because the current grating bank measures one motion
direction. The spectrum is predictive support, not causal attribution; causal
evidence comes from the exact moving-versus-stabilized replay.

`compute_native_rucci_overlap.py` and `analyze_rucci_causal_link.py` are retained
only to reproduce the deprecated instantaneous-velocity diagnostic. Their
`TF=|k dot v(t)|` histogram is not a temporal PSD and must not be used in a
production figure or mechanistic claim.

## Inputs

- checkpoint: `outputs/artifacts/model_checkpoints/fig4_twin/epoch=147-val_bps_overall=0.5702.ckpt`
- frozen dataset config: `paper/fig4/upstream/dataset_configs/multi_basic_120_long.yaml`
- crossed bank: `outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged`
- RR100 audit inputs: the readable final label array and repeated movie-medoid QC
  table in Declan's checkout

The historical RR100 archives in Declan's checkout are not readable by other
users. `reconstruct_rr100_spec.py` reconstructs the scientifically relevant
one-hot transform. The 58 final group rows are repeated identically over 14 QC
movies. The retained-channel label array supplies the 42 singleton channels;
post-hoc compression did not exclude channels. The resulting representative
order is the original exporter order: groups 0–57, then singleton channels in
ascending canonical-channel order. An exact replay of the historical grating
grid checks all 100 SF, TF, and orientation peaks against the cached Figure 4
unit table.

## Run order

The model environment used on this workstation is:

```bash
export PYTHONPATH=/home/jake/repos/DataYatesV1
PY=/home/jake/miniconda3/envs/yatesfv/bin/python
```

Then run:

```bash
python paper/fig4/spatiotemporal_tuning/reconstruct_rr100_spec.py

$PY paper/fig4/spatiotemporal_tuning/run_grating_probe.py \
  --device cuda:0 --frame-batch-size 32

$PY paper/fig4/spatiotemporal_tuning/run_controlled_scaling.py \
  --device cuda:0 --frame-batch-size 32 --trace-batch-size 8

$PY paper/fig4/spatiotemporal_tuning/analyze_joint_tuning.py
```

Both GPU runners are cache-aware. The grating probe checkpoints every stimulus
condition and resumes incomplete tensors. Output is isolated under
`outputs/figures/fig4/spatiotemporal_tuning/`.

## Fixed analysis choices

- Existing continuous SF preference and the displayed `<0.5`/`>=0.5` cpd split
  are retained.
- Probe orientation is the bar/contour axis. The spatial-frequency normal is
  `n=(-sin(theta), cos(theta))`.
- The historical tensor exactly retains the 1.5-s 6 SF × 6 TF × 4 orientation
  protocol. The dense-TF tensor retains the same six SFs and four orientations,
  uses 17 TFs from 0.2 to 51.2 Hz, and extends trials to 3 s.
- Each stimulus is independently lag-embedded, and the first 32 output frames
  are discarded. No recurrent state persists between conditions.
- TF maxima at 0.2 or 51.2 Hz are censored. No fit moves them into the interior.
- Drift-only traces are primary. Projection-based analyses require OSI >= 0.05;
  analyses requiring `v*` also require an interior TF maximum.
- Population curves use 250 hierarchical bootstrap replicates, independently
  resampling units, images, and trajectories with seed 20260810.
- The scalar low/high curve RMS comparison uses only common bins containing at
  least 100 unit×trajectory pairs per group; all bins, including sparse tails,
  remain in the released table and plots.
- The temporal-match spectrum removes the complex phase-signal mean, applies a
  Hann taper, excludes DC, combines direction by weighting `|f|`, and only uses
  power within the measured TF range.
- The controlled experiment uses eight contour images, eight drift-only traces
  spanning raw-path quantiles, and fixed scales
  `{0, .25, .5, .75, 1, 1.5, 2, 3}`.

The final scientific decision and all deviations are recorded in
`analysis/report.md` and `analysis/statistics.json`.
